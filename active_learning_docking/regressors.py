"""Local regression backends and acquired-label-only weighted ensembles.

Weights operate on model means, so a model's internal member count never gives
it extra voting power. The mixture variance includes within-model variance and
between-model disagreement. It is an acquisition heuristic, not a calibrated CI.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import ModelConfig
from .model_registry import canonical_model_name
from .store import PackedFingerprints


def _take(X: Any, indices: np.ndarray) -> Any:
    return X.take(indices) if isinstance(X, PackedFingerprints) else X[indices]


def _write_metadata(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class ForestRegressor:
    """Scikit-learn forests with streamed tree spread and pickle-free restore.

    Checkpoints store acquired training arrays and parameters. Loading refits
    the forest deterministically in the same scikit-learn environment.
    """

    def __init__(self, backend: str, *, params: dict | None = None, seed: int = 42,
                 cpu_threads: int = 1, inference_batch_size: int = 256) -> None:
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

        self.model_type = canonical_model_name(backend)
        cls = {"random_forest": RandomForestRegressor, "extra_trees": ExtraTreesRegressor}[self.model_type]
        self.params = dict(params or {})
        unknown = set(self.params) - set(inspect.signature(cls).parameters)
        if unknown:
            raise ValueError(f"Unknown {backend} parameters: {sorted(unknown)}")
        forbidden = set(self.params) & {"random_state", "n_jobs", "warm_start"}
        if forbidden:
            raise ValueError(f"{backend} parameters are controlled by the experiment: {sorted(forbidden)}")
        self.seed, self.cpu_threads, self.inference_batch_size = seed, cpu_threads, inference_batch_size
        self.estimator = cls(**{"n_estimators": 128, **self.params}, random_state=seed, n_jobs=cpu_threads)
        self.estimator._validate_params()
        self.is_fitted = False
        self.history_: list[dict] = []
        self.best_validation_loss_ = None

    def fit(self, X: Any, y: Any, groups: Any = None) -> ForestRegressor:
        from .model import _features, _targets

        self.is_fitted = False
        X = _features(X)
        self.y_ = _targets(y, len(X)).copy()
        if not len(X):
            raise ValueError("At least one acquired labeled sample is needed")
        self.X_ = np.asarray(X[:], dtype=np.float32)
        self.n_features_in_, self.n_samples_fit_ = self.X_.shape[1], len(X)
        self.estimator.fit(self.X_, self.y_)
        self.train_indices_ = np.arange(len(X), dtype=np.int64)
        self.validation_indices_ = np.empty(0, dtype=np.int64)
        self.history_ = [{"model": self.model_type, "n_samples": len(X)}]
        self.is_fitted = True
        return self

    def predict(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        from .model import _features

        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features; expected {self.n_features_in_}")
        mean, std = np.empty(len(X)), np.empty(len(X))
        for start in range(0, len(X), self.inference_batch_size):
            batch = np.asarray(X[start:start + self.inference_batch_size], dtype=np.float32)
            avg, m2 = np.zeros(len(batch)), np.zeros(len(batch))
            for count, tree in enumerate(self.estimator.estimators_, 1):
                prediction = tree.predict(batch)
                delta = prediction - avg
                avg += delta / count
                m2 += delta * (prediction - avg)
            mean[start:start + len(batch)] = avg
            std[start:start + len(batch)] = np.sqrt(np.maximum(m2 / count, 0))
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise RuntimeError(f"{self.model_type} produced non-finite predictions")
        return mean, std

    def get_model_info(self) -> dict:
        import sklearn

        return {"model_type": self.model_type, "implementation": type(self.estimator).__name__,
                "sklearn_version": sklearn.__version__, "parameters": self.params,
                "seed": self.seed, "cpu_threads": self.cpu_threads,
                "inference_batch_size": self.inference_batch_size, "is_fitted": self.is_fitted,
                "n_samples_fit": getattr(self, "n_samples_fit_", 0),
                "uncertainty": "tree_population_standard_deviation",
                "restore": "deterministic_refit_from_acquired_training_arrays"}

    def save(self, directory: str | Path) -> Path:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        data = directory / "training_data.npz"
        np.savez_compressed(data, X=self.X_, y=self.y_)
        _write_metadata(directory / "model_metadata.json", {
            "checkpoint_version": 1, "model_info": self.get_model_info(),
            "training_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        })
        return directory

    @classmethod
    def load(cls, directory: str | Path, *, device: str = "cpu") -> ForestRegressor:
        import sklearn

        directory = Path(directory)
        meta = json.loads((directory / "model_metadata.json").read_text())
        info = meta["model_info"]
        if meta.get("checkpoint_version") != 1 or info["sklearn_version"] != sklearn.__version__:
            raise ValueError("Forest restore requires the original checkpoint and scikit-learn version")
        data = directory / "training_data.npz"
        if hashlib.sha256(data.read_bytes()).hexdigest() != meta["training_sha256"]:
            raise ValueError("Checkpoint SHA256 mismatch")
        estimator = cls(info["model_type"], params=info["parameters"], seed=info["seed"],
                        cpu_threads=info["cpu_threads"], inference_batch_size=info["inference_batch_size"])
        with np.load(data, allow_pickle=False) as arrays:
            estimator.fit(arrays["X"], arrays["y"])
        return estimator


def _make_member(name: str, config: ModelConfig, seed: int, device: str, cpu_threads: int) -> Any:
    params = dict(config.params.get(name, {}))
    batch = config.inference_batch_size or config.batch_size
    if name == "tabm":
        from .model import TabMRegressor

        defaults = {key: value for key, value in asdict(config).items()
                    if key not in {"models", "params", "ensemble"}}
        unknown = set(params) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown tabm parameters: {sorted(unknown)}")
        effective = {**defaults, **params}
        if type(effective["k"]) is not int or effective["k"] < 2:
            raise ValueError("tabm.k must be at least 2 for ensemble uncertainty")
        if effective["auto_tune"]:
            try:
                import optuna  # noqa: F401
            except ImportError as exc:
                raise ImportError("TabM auto_tune requires pip install -e '.[tuning]'") from exc
        return TabMRegressor(**effective, seed=seed, device=device)
    if name in {"random_forest", "extra_trees"}:
        return ForestRegressor(name, params=params, seed=seed, cpu_threads=cpu_threads,
                               inference_batch_size=batch)
    from .foundation import FoundationRegressor

    return FoundationRegressor(name, params=params, seed=seed, device=device,
                               cpu_threads=cpu_threads, inference_batch_size=batch)


def create_regressor(config: ModelConfig, *, seed: int = 42, device: str = "cpu", cpu_threads: int = 1) -> Any:
    """Construct only explicitly selected models; the legacy default is TabM."""
    config.validate_selection()
    if len(config.models) == 1:
        return _make_member(config.models[0], config, seed, device, cpu_threads)
    return RegressionEnsemble(config, seed=seed, device=device, cpu_threads=cpu_threads)


class RegressionEnsemble:
    """Sequential model fits with fixed or leakage-safe validation weights."""

    def __init__(self, config: ModelConfig, *, seed: int = 42, device: str = "cpu", cpu_threads: int = 1):
        config.validate_selection()
        self.config, self.seed, self.device, self.cpu_threads = config, seed, device, cpu_threads
        self.model_type = "ensemble"
        self.members_ = self._new_members()
        self.is_fitted = False
        self.history_: list[dict] = []
        self.best_validation_loss_ = None

    def _new_members(self) -> dict[str, Any]:
        return {name: _make_member(name, self.config, (self.seed + index) % (2**32), self.device,
                                  self.cpu_threads) for index, name in enumerate(self.config.models)}

    def preflight(self) -> None:
        """Verify optional local runtimes/weights before oracle calls."""
        for member in self.members_.values():
            if hasattr(member, "preflight"):
                member.preflight()

    def _weight_split(self, n: int, groups: Any) -> tuple[np.ndarray, np.ndarray]:
        if groups is None:
            groups = np.asarray([str(i) for i in range(n)], dtype=object)
        else:
            groups = np.asarray(groups, dtype=object)
        if groups.shape != (n,) or any(not isinstance(g, str) or not g for g in groups):
            raise ValueError("groups must contain one nonempty string identifier per row")
        grouped_rows: dict[str, list[int]] = {}
        for index, group in enumerate(groups):
            grouped_rows.setdefault(group, []).append(index)
        keys = sorted(grouped_rows)
        min_train = 2 if any(name not in {"random_forest", "extra_trees"} for name in self.config.models) else 1
        if len(keys) <= min_train:
            raise ValueError(f"Validation ensemble requires at least {min_train + 1} distinct acquired groups")
        order = np.random.default_rng(self.seed).permutation(len(keys))
        target = max(1, int(np.ceil(n * self.config.validation_fraction)))
        validation = []
        for index in order[:-min_train]:
            validation.extend(grouped_rows[keys[index]])
            if len(validation) >= target:
                break
        valid = np.asarray(sorted(validation), dtype=np.int64)
        return np.setdiff1d(np.arange(n), valid), valid

    def fit(self, X: Any, y: Any, groups: Any = None) -> RegressionEnsemble:
        from .model import _features, _targets

        self.is_fitted = False
        X = _features(X)
        y = _targets(y, len(X))
        if len(X) < 2:
            raise ValueError("At least two acquired labeled samples are needed")
        groups = None if groups is None else np.asarray(groups, dtype=object)
        self.n_features_in_, self.n_samples_fit_ = X.shape[1], len(X)
        self.weight_train_indices_, self.weight_validation_indices_ = np.arange(len(X)), np.empty(0, dtype=int)
        self.validation_mse_: dict[str, float] = {}
        strategy = self.config.ensemble.strategy
        if strategy == "validation":
            train, valid = self._weight_split(len(X), groups)
            self.weight_train_indices_, self.weight_validation_indices_ = train, valid
            errors = []
            # Probe models see only the training part, including their own tuning.
            for name in self.config.models:
                member = self.members_[name]
                member.fit(_take(X, train), y[train], groups=None if groups is None else groups[train])
                mean, _ = member.predict(_take(X, valid))
                error = float(np.mean((mean.astype(np.longdouble) - y[valid]) ** 2))
                if not np.isfinite(error):
                    raise RuntimeError(f"Non-finite ensemble validation MSE from {name}")
                errors.append(error)
                self.validation_mse_[name] = error
                # Release large pretrained contexts/caches before refitting.
                self.members_[name] = None
                del member
            errors = np.maximum(errors, 1e-12)
            weights = np.min(errors) / errors
            self.members_ = self._new_members()
        elif strategy == "weighted":
            weights = np.asarray(self.config.ensemble.weights, dtype=np.float64)
        else:
            weights = np.ones(len(self.config.models))
        # Scale before summing to avoid overflow with large user weights.
        weights = weights / np.max(weights)
        self.weights_ = weights / weights.sum()
        self.history_ = []
        for name, member in self.members_.items():
            member.fit(X, y, groups=groups)
            self.history_.extend({**row, "model": name} for row in member.history_)
        self.train_indices_, self.validation_indices_ = np.arange(len(X)), np.empty(0, dtype=np.int64)
        self.is_fitted = True
        return self

    def predict(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        from .model import _features

        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features; expected {self.n_features_in_}")
        output_mean, output_std = np.empty(len(X)), np.empty(len(X))
        batch_size = self.config.inference_batch_size or self.config.batch_size
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size]
            means, variances, weights = [], [], []
            for weight, member in zip(self.weights_, self.members_.values()):
                if weight == 0:
                    continue
                mean, std = member.predict(batch)
                mean, std = np.asarray(mean), np.asarray(std)
                if mean.shape != (len(batch),) or std.shape != mean.shape or (std < 0).any():
                    raise RuntimeError("Ensemble member returned invalid prediction shape or uncertainty")
                if not np.isfinite(mean).all() or not np.isfinite(std).all():
                    raise RuntimeError("Ensemble member returned non-finite predictions")
                means.append(mean)
                variances.append(std ** 2)
                weights.append(weight)
            means, weights = np.asarray(means), np.asarray(weights)
            mean = np.sum(weights[:, None] * means, axis=0)
            # Center first: E[var] + E[(mean_i - mean)^2] avoids cancellation.
            variance = np.sum(weights[:, None] * (np.asarray(variances) + (means - mean) ** 2), axis=0)
            output_mean[start:start + len(batch)] = mean
            output_std[start:start + len(batch)] = np.sqrt(np.maximum(variance, 0))
        if not np.isfinite(output_mean).all() or not np.isfinite(output_std).all():
            raise RuntimeError("Ensemble produced non-finite predictions")
        return output_mean, output_std

    def get_model_info(self) -> dict:
        return {"model_type": "ensemble", "models": self.config.models, "is_fitted": self.is_fitted,
                "strategy": self.config.ensemble.strategy,
                "weights": dict(zip(self.config.models, self.weights_.tolist())) if self.is_fitted else None,
                "validation_mse": getattr(self, "validation_mse_", {}),
                "weight_selection": "acquired_group_validation_then_refit" if
                    self.config.ensemble.strategy == "validation" else "fixed_weights",
                "uncertainty": "mixture_population_standard_deviation",
                "n_samples_fit": getattr(self, "n_samples_fit_", 0),
                "members": {name: model.get_model_info() for name, model in self.members_.items()}}

    def save(self, directory: str | Path) -> Path:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, member in self.members_.items():
            member.save(directory / name)
        _write_metadata(directory / "model_metadata.json", {
            "checkpoint_version": 1, "model_info": self.get_model_info(), "config": asdict(self.config),
            "seed": self.seed, "cpu_threads": self.cpu_threads, "n_features": self.n_features_in_,
            "weight_train_indices": self.weight_train_indices_.tolist(),
            "weight_validation_indices": self.weight_validation_indices_.tolist(), "history": self.history_,
        })
        return directory

    @classmethod
    def load(cls, directory: str | Path, *, device: str = "cpu") -> RegressionEnsemble:
        from .config import _construct

        directory = Path(directory)
        meta = json.loads((directory / "model_metadata.json").read_text())
        if meta.get("checkpoint_version") != 1:
            raise ValueError("Unsupported ensemble checkpoint version")
        config = _construct(ModelConfig, meta["config"])
        estimator = cls(config, seed=meta["seed"], device=device, cpu_threads=meta["cpu_threads"])
        estimator.members_ = {name: load_regressor(directory / name, device=device) for name in config.models}
        estimator.weights_ = np.array([meta["model_info"]["weights"][name] for name in config.models])
        if (not np.isfinite(estimator.weights_).all() or (estimator.weights_ < 0).any()
                or not np.isclose(estimator.weights_.sum(), 1)):
            raise ValueError("Invalid saved ensemble weights")
        estimator.n_features_in_ = meta["n_features"]
        estimator.n_samples_fit_ = meta["model_info"]["n_samples_fit"]
        estimator.train_indices_, estimator.validation_indices_ = np.arange(estimator.n_samples_fit_), np.empty(0, int)
        estimator.weight_train_indices_ = np.asarray(meta["weight_train_indices"], dtype=np.int64)
        estimator.weight_validation_indices_ = np.asarray(meta["weight_validation_indices"], dtype=np.int64)
        estimator.validation_mse_ = meta["model_info"]["validation_mse"]
        estimator.history_ = meta["history"]
        estimator.is_fitted = True
        return estimator


def load_regressor(directory: str | Path, *, device: str = "cpu") -> Any:
    """Restore any supported checkpoint, including existing TabM directories."""
    meta = json.loads((Path(directory) / "model_metadata.json").read_text())
    name = meta["model_info"]["model_type"]
    if name == "ensemble":
        return RegressionEnsemble.load(directory, device=device)
    if name == "tabm":
        from .model import TabMRegressor
        return TabMRegressor.load(directory, device=device)
    if name in {"random_forest", "extra_trees"}:
        return ForestRegressor.load(directory, device=device)
    from .foundation import FoundationRegressor
    return FoundationRegressor.load(directory, device=device)
