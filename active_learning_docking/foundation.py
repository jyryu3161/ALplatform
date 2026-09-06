"""Local, optional tabular foundation regressors with bounded pool inference.

Only acquired labels enter the context. Context rows are sampled reproducibly
before variance-based feature selection; query rows never fit preprocessing.
Nori and TabLDM expose quantile-bank spread (an approximation to predictive
standard deviation). Google's public TabFM API exposes point estimates only.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution, version
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np


FOUNDATION_BACKENDS = ("nori", "tabldm", "tabfm")
_PACKAGES = {"nori": "synthefy-nori", "tabldm": "Xiaomi-TabLDM", "tabfm": "tabfm"}
_REPOSITORIES = {
    "nori": "https://github.com/Synthefy/synthefy-nori",
    "tabldm": "https://github.com/xiaomi-research/xiaomi-tabldm",
    "tabfm": "https://github.com/google-research/tabfm",
}
_NORI_CHECKPOINTS = {
    "nori-6m": ("Synthefy/Nori", "1aca005143362552bb56c62e110ab77767cf4f77"),
    "nori-30m": ("Synthefy/Nori-30M", "63c9f7facf9fb32c37ce3fc2fba331d524696318"),
    "nori-100m": ("Synthefy/Nori-100M", "467ae9704fb0ab328c0d2f850602c27123928b48"),
}
_COMMON = {"max_features", "max_context_samples"}
_SUPPORTED = {
    "nori": {"model", "model_path", "augmentations", "svd_dim"},
    "tabldm": {
        "model_path", "allow_auto_download", "checkpoint_version", "n_estimators",
        "batch_size", "kv_cache", "norm_methods", "feat_shuffle_method", "use_amp", "offload_mode",
        "disk_offload_dir",
    },
    "tabfm": {
        "checkpoint_path", "n_estimators", "batch_size", "max_num_rows", "norm_methods",
    },
}


def validate_foundation_params(backend: str, params: dict[str, Any] | None) -> None:
    """Validate the documented integration options without importing PyTorch."""
    if backend not in FOUNDATION_BACKENDS:
        raise ValueError(f"Unknown foundation backend {backend!r}; choose {FOUNDATION_BACKENDS}")
    if params is None:
        return
    if not isinstance(params, dict):
        raise ValueError(f"{backend} params must be a mapping")
    if any(not isinstance(key, str) for key in params):
        raise ValueError(f"{backend} parameter names must be strings")
    unknown = set(params) - _COMMON - _SUPPORTED[backend]
    if unknown:
        raise ValueError(f"Unknown {backend} parameters: {sorted(unknown)}")
    for key in ("max_features", "max_context_samples", "max_num_rows", "svd_dim", "n_estimators", "batch_size"):
        if key not in params:
            continue
        value = params[key]
        if value is None and key in {"max_features", "max_context_samples", "max_num_rows", "svd_dim"}:
            continue
        minimum = 2 if key in {"max_context_samples", "max_num_rows"} else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{backend}.{key} must be an integer >= {minimum}" +
                             (" or null" if key in _COMMON | {"max_num_rows", "svd_dim"} else ""))
    for key in ("allow_auto_download",):
        if key in params and type(params[key]) is not bool:
            raise ValueError(f"{backend}.{key} must be boolean")
    for key in ("model_path", "checkpoint_path", "checkpoint_version", "disk_offload_dir"):
        if key in params and (not isinstance(params[key], str) or not params[key].strip()):
            raise ValueError(f"{backend}.{key} must be a nonempty path/name string")
    if backend == "nori":
        if "model" in params and (not isinstance(params["model"], str) or
                                  params["model"] not in {"nori-6m", "nori-30m", "nori-100m"}):
            raise ValueError("nori.model must be nori-6m, nori-30m, or nori-100m")
        if "augmentations" in params and (not isinstance(params["augmentations"], (list, tuple)) or
                                          any(not isinstance(v, str) or v != "yj" for v in params["augmentations"])):
            raise ValueError("nori.augmentations must be [] or ['yj']")
    if "kv_cache" in params and type(params["kv_cache"]) is not bool and params["kv_cache"] not in ("kv", "repr"):
        raise ValueError("tabldm.kv_cache must be boolean, 'kv', or 'repr'")
    if "use_amp" in params and type(params["use_amp"]) is not bool and params["use_amp"] != "auto":
        raise ValueError("tabldm.use_amp must be boolean or 'auto'")
    if "offload_mode" in params and type(params["offload_mode"]) is not bool and (
            not isinstance(params["offload_mode"], str) or params["offload_mode"] not in {"auto", "cpu", "disk", "gpu"}):
        raise ValueError("tabldm.offload_mode must be boolean, auto, cpu, disk, or gpu")
    if params.get("offload_mode") == "disk" and not params.get("disk_offload_dir"):
        raise ValueError("tabldm.offload_mode='disk' requires disk_offload_dir")
    if "feat_shuffle_method" in params and (not isinstance(params["feat_shuffle_method"], str) or
                                            params["feat_shuffle_method"] not in {"latin", "random", "none", "shift"}):
        raise ValueError("tabldm.feat_shuffle_method must be latin, random, none, or shift")
    if "norm_methods" in params:
        methods = params["norm_methods"]
        allowed = {"none", "power", "quantile", "quantile_rtdl", "robust"}
        methods = [methods] if isinstance(methods, str) else methods
        if not isinstance(methods, list) or not methods or any(not isinstance(v, str) or v not in allowed
                                                             for v in methods):
            raise ValueError(f"{backend}.norm_methods must be a nonempty list of supported transforms")
    # Config and saved metadata must remain JSON values, without serialized objects.
    try:
        json.dumps(params, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{backend} params must contain finite JSON values") from exc


class FoundationRegressor:
    """Consistent fit/predict/save interface around the official local packages.

    ``max_context_samples`` bounds acquired context (default 2048), and
    ``max_features`` retains the 128 highest-variance features from that context.
    Set either to ``None`` to retain all. Prediction materializes at most
    ``inference_batch_size`` pool rows. No hosted client or API key is used.
    """

    CHECKPOINT_VERSION = 2

    def __init__(self, backend: str, *, params: dict[str, Any] | None = None,
                 seed: int = 42, device: str = "cpu", cpu_threads: int = 1,
                 inference_batch_size: int = 256) -> None:
        validate_foundation_params(backend, params)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name, value in (("cpu_threads", cpu_threads), ("inference_batch_size", inference_batch_size)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if device not in {"cpu", "cuda:0", "cuda:1"}:
            raise ValueError("device must be cpu, cuda:0, or cuda:1 after resource configuration")
        if backend == "tabfm" and sys.version_info < (3, 11):
            raise RuntimeError("Google TabFM requires Python >= 3.11; create a Python 3.11 environment")
        self.backend = self.model_type = backend
        self.params = dict(params or {})
        self.seed, self.device = seed, device
        self.cpu_threads, self.inference_batch_size = cpu_threads, inference_batch_size
        self.max_features = self.params.get("max_features", 128)
        self.max_context_samples = self.params.get("max_context_samples", 2048)
        self.is_fitted = False
        self.model = None
        self.history_: list[dict[str, Any]] = []
        self.validation_indices_ = np.empty(0, dtype=np.int64)
        self.best_validation_loss_ = None
        self.checkpoint_reference_: dict[str, Any] | None = None
        self._validate_device_mapping()

    def _validate_device_mapping(self) -> None:
        """Match the physical GPU 0/1 restriction even for direct API callers."""
        if self.device == "cpu":
            return
        mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        if mask is not None:
            tokens = [token.strip() for token in mask.split(",") if token.strip()]
            if any(token not in {"0", "1"} for token in tokens) or len(tokens) != len(set(tokens)):
                raise ValueError("CUDA_VISIBLE_DEVICES must contain unique numeric physical GPU IDs 0 or 1")
            if int(self.device.split(":")[1]) >= len(tokens):
                raise ValueError(f"Requested logical {self.device} is excluded by CUDA_VISIBLE_DEVICES")

    @contextmanager
    def _runtime(self):
        self._validate_device_mapping()
        import torch
        from threadpoolctl import threadpool_limits

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {self.device}, but CUDA is unavailable")
        devices = [int(self.device.split(":")[1])] if self.device.startswith("cuda") else []
        previous_threads = torch.get_num_threads()
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            torch.set_num_threads(self.cpu_threads)
            np.random.seed(self.seed)
            random.seed(self.seed)
            with torch.random.fork_rng(devices=devices), threadpool_limits(limits=self.cpu_threads):
                torch.random.default_generator.manual_seed(self.seed)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(self.seed)
                with torch.inference_mode():
                    yield
        finally:
            torch.set_num_threads(previous_threads)
            np.random.set_state(numpy_state)
            random.setstate(python_state)

    def preflight(self) -> None:
        """Check the optional package/API and local paths before acquiring labels.

        Public weights are downloaded and hashed before docking starts. This
        avoids constructing the neural network and needs no labeled context.
        """
        self._validate_device_mapping()
        module_name, class_name = {
            "nori": ("synthefy_nori", "NoriRegressor"),
            "tabldm": ("tabldm", "TabLDMRegressor"),
            "tabfm": ("tabfm", "TabFMRegressor"),
        }[self.backend]
        try:
            module = import_module(module_name)
            getattr(module, class_name)
            if self.backend == "tabfm":
                getattr(module.tabfm_v1_0_0_pytorch, "load")
                import_module("safetensors.torch")
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"{self.backend} requires the supported local {_PACKAGES[self.backend]} package; "
                f"install the project's '{self.backend}' extra (see README). No model fallback is used."
            ) from exc
        for key in ("model_path", "checkpoint_path"):
            if key in self.params:
                path = Path(self.params[key])
                exists = path.is_dir() if key == "checkpoint_path" else path.is_file()
                if not exists:
                    raise FileNotFoundError(f"{self.backend}.{key} does not exist: {path}")
        self._resolve_checkpoint({key: value for key, value in self.params.items() if key not in _COMMON})

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _resolve_checkpoint(self, options: dict[str, Any]) -> dict[str, Any]:
        """Resolve public immutable Hub revisions or explicit local checkpoints."""
        key = "checkpoint_path" if self.backend == "tabfm" else "model_path"
        if self.checkpoint_reference_ is not None:
            options[key] = self.checkpoint_reference_["path"]
            return options
        repo, revision = None, None
        if key in options:
            path = Path(options[key]).expanduser().resolve()
        else:
            from huggingface_hub import hf_hub_download, snapshot_download

            if self.backend == "nori":
                repo, revision = _NORI_CHECKPOINTS[options.get("model", "nori-6m")]
                path = Path(hf_hub_download(repo, "nori.pt", revision=revision, token=False))
            elif self.backend == "tabldm":
                repo, revision = "occams/Xiaomi-TabLDM", "0e10427273bc862fb60fe7df84c89e5298e16cc7"
                path = Path(hf_hub_download(
                    repo, options.get("checkpoint_version", "checkpoints/reg_default.ckpt"),
                    revision=revision, token=False, local_files_only=not options.get("allow_auto_download", True),
                ))
            else:
                repo, revision = "google/tabfm-1.0.0-pytorch", "77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83"
                path = Path(snapshot_download(repo, revision=revision, token=False,
                                              allow_patterns=["regression/config.json", "regression/model.safetensors"]))
        if self.backend == "tabfm":
            if (path / "regression").is_dir():
                path = path / "regression"
            files = [path / "config.json"]
            weights = path / "model.safetensors"
            if not weights.is_file():
                raise FileNotFoundError(
                    f"TabFM local checkpoint requires config.json and model.safetensors in {path}; "
                    "the supported PyTorch loader does not load local pytorch_model.bin files"
                )
            files.append(weights)
        else:
            files = [path]
        for file in files:
            if not file.is_file():
                raise FileNotFoundError(f"{self.backend} checkpoint file does not exist: {file}")
        self.checkpoint_reference_ = {
            "path": str(path.absolute()), "repository": repo, "revision": revision,
            "files": [{"path": str(file.absolute()), "sha256": self._file_hash(file),
                       "hub_filename": (str(file).split(f"/snapshots/{revision}/", 1)[1] if repo else None)}
                      for file in files],
        }
        options[key] = self.checkpoint_reference_["path"]
        return options

    def _make_model(self):
        options = {key: value for key, value in self.params.items() if key not in _COMMON}
        try:
            options = self._resolve_checkpoint(options)
            if self.backend == "nori":
                cls = import_module("synthefy_nori").NoriRegressor
                options.setdefault("model", "nori-6m")
                return cls(device=self.device, token=False, large_context_seed=self.seed, **options)
            if self.backend == "tabldm":
                cls = import_module("tabldm").TabLDMRegressor
                options.setdefault("n_estimators", 8)
                # Upstream's positive n_jobs erroneously expands to all CPU cores.
                # Our enclosing runtime bounds Torch and BLAS threads instead.
                return cls(device=self.device, random_state=self.seed, n_jobs=None, **options)
            module = import_module("tabfm")
            # Google's PyTorch extra omits this required Hub loader dependency.
            # Check it before allocating the model, including direct fit callers.
            import_module("safetensors.torch")
            checkpoint = options.pop("checkpoint_path", None)
            base = module.tabfm_v1_0_0_pytorch.load(
                model_type="regression", checkpoint_path=checkpoint, device=self.device,
            )
            options.setdefault("n_estimators", 8)
            options.setdefault("max_num_features", self.max_features)
            return module.TabFMRegressor(model=base, random_state=self.seed, **options)
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"{self.backend} requires the supported local {_PACKAGES[self.backend]} package; "
                f"install the project's '{self.backend}' extra (see README). No model fallback is used."
            ) from exc

    def fit(self, X: Any, y: Any, groups: Any = None) -> FoundationRegressor:
        from .model import _features, _targets

        X, y = _features(X), _targets(y, len(X))
        if len(X) < 2:
            raise ValueError("At least two acquired labeled samples are needed for foundation context")
        if groups is not None:
            values = np.asarray(groups, dtype=object)
            if values.shape != (len(X),) or any(not isinstance(v, str) or not v for v in values):
                raise ValueError("groups must contain one nonempty string identifier per row")
        self.is_fitted = False
        self.n_features_in_, self.n_samples_fit_ = X.shape[1], len(X)
        count = min(len(X), self.max_context_samples or len(X))
        self.train_indices_ = np.sort(np.random.default_rng(self.seed).choice(len(X), count, replace=False))
        context = np.asarray(X[self.train_indices_], dtype=np.float32)
        variances = context.var(axis=0, dtype=np.float64)
        if not np.any(variances > 0):
            raise ValueError(
                f"{self.backend} needs at least one varying fingerprint feature in the selected context; "
                "increase max_context_samples/fingerprint bits or acquire more diverse molecules"
            )
        feature_count = min(X.shape[1], self.max_features or X.shape[1])
        # Stable sort breaks variance ties by original feature index.
        self.feature_indices_ = np.sort(np.argsort(-variances, kind="stable")[:feature_count])
        self.context_X_ = np.ascontiguousarray(context[:, self.feature_indices_])
        self.context_y_ = y[self.train_indices_].copy()
        self._fit_context()
        self.history_ = [{"model_type": self.backend, "n_context": count,
                          "n_features_selected": feature_count, "weight_updates": 0}]
        self.is_fitted = True
        return self

    def _fit_context(self) -> None:
        with self._runtime():
            self.model = self._make_model()
            if self.backend == "tabldm":
                # This upstream version reads the fit argument, ignoring the
                # constructor's identically named setting during plain fitting.
                # Its plain fit also omits this auxiliary group's initialization
                # although _build_kv_cache reads it unconditionally (0.1.0).
                self.model.quantile_ensemble_generator_ = None
                self.model.fit(self.context_X_, self.context_y_, kv_cache=self.params.get("kv_cache", False))
            else:
                self.model.fit(self.context_X_, self.context_y_)

    def _require_fitted(self) -> None:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("FoundationRegressor is not fitted; call fit or load first")

    @staticmethod
    def _quantile_std(quantiles: Any, n_rows: int, levels: Any = None) -> np.ndarray:
        quantiles = np.asarray(quantiles, dtype=np.float64)
        if quantiles.ndim != 2 or quantiles.shape[0] != n_rows or quantiles.shape[1] < 2:
            raise RuntimeError("Foundation backend returned an invalid quantile matrix")
        if not np.isfinite(quantiles).all():
            raise RuntimeError("Foundation backend returned non-finite quantiles")
        if levels is None:
            return quantiles.std(axis=1, ddof=0)
        levels = np.asarray(levels, dtype=np.float64)
        if levels.shape != (quantiles.shape[1],) or not np.isfinite(levels).all() or \
                np.any(np.diff(levels) <= 0) or levels[0] <= 0 or levels[-1] >= 1:
            raise RuntimeError("Foundation backend returned invalid quantile levels")
        # Integrate a nearest-quantile step function; tails use endpoint values.
        edges = np.r_[0.0, (levels[:-1] + levels[1:]) / 2, 1.0]
        weights = np.diff(edges)
        mean = quantiles @ weights
        return np.sqrt(((quantiles - mean[:, None]) ** 2) @ weights)

    def predict(self, X: Any, return_std: bool = True):
        from .model import _features

        self._require_fitted()
        X = _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features; expected {self.n_features_in_}")
        mean, std = np.empty(len(X)), np.zeros(len(X))
        if len(X):
            with self._runtime():
                for start in range(0, len(X), self.inference_batch_size):
                    stop = min(len(X), start + self.inference_batch_size)
                    batch = np.asarray(X[start:stop], dtype=np.float32)[:, self.feature_indices_]
                    if self.backend == "nori":
                        result = self.model.predict(batch, output_type="full")
                        prediction = result["mean"]
                        std[start:stop] = self._quantile_std(result["quantiles"], len(batch), result["taus"])
                    elif self.backend == "tabldm":
                        # Do not use upstream output_type='variance': its sklearn
                        # wrapper applies the target inverse transform to variance.
                        result = self.model.predict(batch, output_type=["mean", "raw_quantiles"])
                        prediction = result["mean"]
                        std[start:stop] = self._quantile_std(result["raw_quantiles"], len(batch))
                    else:
                        prediction = self.model.predict(batch)
                    prediction = np.asarray(prediction, dtype=np.float64)
                    if prediction.shape != (len(batch),):
                        raise RuntimeError(f"{self.backend} returned predictions with invalid shape {prediction.shape}")
                    mean[start:stop] = prediction
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise RuntimeError(f"{self.backend} returned non-finite predictions or uncertainty")
        return (mean, std) if return_std else mean

    def get_model_info(self) -> dict[str, Any]:
        try:
            package_version = version(_PACKAGES[self.backend])
        except PackageNotFoundError:
            package_version = None
        info = {
            "model_type": self.backend, "implementation": {
                "nori": "synthefy_nori.NoriRegressor", "tabldm": "tabldm.TabLDMRegressor",
                "tabfm": "tabfm.TabFMRegressor (PyTorch)",
            }[self.backend],
            "upstream_repository": _REPOSITORIES[self.backend], "package_version": package_version,
            "implementation_revision": self._implementation_revision(),
            "is_fitted": self.is_fitted, "execution": "local_in_context_learning",
            "parameters": self.params, "seed": self.seed, "device": self.device,
            "cpu_threads": self.cpu_threads, "inference_batch_size": self.inference_batch_size,
            "max_context_samples": self.max_context_samples, "max_features": self.max_features,
            "uncertainty": "unavailable_zero" if self.backend == "tabfm" else "quantile_bank_standard_deviation",
            "uncertainty_is_calibrated": False,
            "checkpoint_reference": self.checkpoint_reference_,
        }
        if self.is_fitted:
            info.update(n_features=self.n_features_in_, n_samples_fit=self.n_samples_fit_,
                        n_training=len(self.train_indices_), n_validation=0,
                        n_features_selected=len(self.feature_indices_),
                        feature_selection="context_only_variance", context_sampling="seeded_without_replacement")
        return info

    def _implementation_revision(self) -> str | None:
        """Record the actual Git commit for source installs sharing a version."""
        try:
            direct = distribution(_PACKAGES[self.backend]).read_text("direct_url.json")
        except PackageNotFoundError:
            return None
        return json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None

    def save(self, directory: str | Path) -> Path:
        """Persist numeric context and selection, never pickle executable objects.

        Pretrained weights remain in the upstream download cache or configured
        local checkpoint. Restoring replays the exact selected context.
        """
        self._require_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        context_path = directory / "foundation_context.npz"
        np.savez_compressed(context_path, X=self.context_X_, y=self.context_y_,
                            train_indices=self.train_indices_, feature_indices=self.feature_indices_)
        metadata = {"checkpoint_version": self.CHECKPOINT_VERSION,
                    "context_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
                    "model_info": self.get_model_info(), "history": self.history_}
        (directory / "model_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        return directory

    @classmethod
    def load(cls, directory: str | Path, *, device: str = "cpu") -> FoundationRegressor:
        directory = Path(directory)
        metadata = json.loads((directory / "model_metadata.json").read_text())
        if metadata.get("checkpoint_version") != cls.CHECKPOINT_VERSION:
            raise ValueError("Unsupported foundation checkpoint version")
        path = directory / "foundation_context.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["context_sha256"]:
            raise ValueError("Foundation context SHA256 mismatch")
        info = metadata["model_info"]
        estimator = cls(info["model_type"], params=info["parameters"], seed=info["seed"], device=device,
                        cpu_threads=info["cpu_threads"], inference_batch_size=info["inference_batch_size"])
        try:
            installed_version = version(_PACKAGES[estimator.backend])
        except PackageNotFoundError:
            installed_version = None
        if installed_version != info["package_version"]:
            raise ValueError(f"{estimator.backend} package version mismatch: checkpoint requires "
                             f"{info['package_version']}, installed {installed_version}")
        if estimator._implementation_revision() != info.get("implementation_revision"):
            raise ValueError(f"{estimator.backend} installed source revision does not match the saved checkpoint")
        estimator.checkpoint_reference_ = info["checkpoint_reference"]
        if not estimator.checkpoint_reference_:
            raise ValueError("Foundation checkpoint has no pretrained weight reference")
        reference = estimator.checkpoint_reference_
        # TabFM needs config and weights in one directory. If either disappeared,
        # resolve the entire public checkpoint into the current cache together.
        relocate = bool(reference.get("repository") and reference.get("revision") and
                        any(not Path(entry["path"]).is_file() for entry in reference["files"]))
        for entry in reference["files"]:
            file = Path(entry["path"])
            if relocate and entry.get("hub_filename"):
                from huggingface_hub import hf_hub_download

                file = Path(hf_hub_download(reference["repository"], entry["hub_filename"],
                                           revision=reference["revision"], token=False))
                entry["path"] = str(file.absolute())
            if not file.is_file():
                raise FileNotFoundError(f"Restore the referenced pretrained weight/cache file: {file}")
            if estimator._file_hash(file) != entry["sha256"]:
                raise ValueError(f"Pretrained checkpoint SHA256 mismatch: {file}")
        # A copied run can rehydrate the same immutable public weights in a
        # different Hugging Face cache while preserving the recorded hashes.
        first_file = Path(estimator.checkpoint_reference_["files"][0]["path"])
        estimator.checkpoint_reference_["path"] = str(first_file.parent if estimator.backend == "tabfm"
                                                     else first_file)
        with np.load(path, allow_pickle=False) as arrays:
            estimator.context_X_, estimator.context_y_ = arrays["X"], arrays["y"]
            estimator.train_indices_, estimator.feature_indices_ = arrays["train_indices"], arrays["feature_indices"]
        estimator.n_features_in_, estimator.n_samples_fit_ = info["n_features"], info["n_samples_fit"]
        estimator.history_ = metadata["history"]
        estimator._fit_context()
        estimator.is_fitted = True
        return estimator
