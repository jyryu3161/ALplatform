"""Reproducible regression using the official :mod:`tabm` implementation.

Optimize the mean of member losses; evaluate the mean prediction. Uncertainty is
the population spread of correlated members, not calibrated predictive error.
Configure resource limits before importing this module (and PyTorch).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tabm import TabM

from .store import PackedFingerprints


_INTEGER_DTYPES = frozenset(np.dtype(kind) for kind in (np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8))


def _features(values: Any) -> np.ndarray | PackedFingerprints:
    """Validate a feature matrix without expanding compact storage.

    :class:`PackedFingerprints` views stay lazy and are unpacked one batch at a
    time, so a memory-mapped pool never materializes. Boolean and integer
    matrices (for example ``uint8`` fingerprints) are kept as they are; other
    inputs are converted to a finite ``float32`` matrix up front.
    """
    if isinstance(values, PackedFingerprints):
        if values.shape[1] == 0:
            raise ValueError("X must have at least one feature")
        return values
    try:
        array = np.asarray(values)
        if array.dtype.kind in "biu":
            if array.dtype not in _INTEGER_DTYPES:
                array = array.astype(np.int64)
        else:
            array = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a rectangular numeric matrix") from exc
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("X must have shape (n_samples, n_features), with n_features > 0")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("X must contain only finite values representable as float32")
    return np.ascontiguousarray(array)


def _batch(array: np.ndarray, device: str) -> torch.Tensor:
    """Move one batch to the model device as ``float32``."""
    return torch.from_numpy(array).to(device, torch.float32)


def _targets(values: Any, n_samples: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or len(array) != n_samples:
        raise ValueError("y must contain one scalar target per row of X")
    if not np.isfinite(array).all():
        raise ValueError("y must contain only finite values")
    return array


class TabMRegressor:
    """Numerical-feature regressor backed by ``tabm.TabM.make``.

    ``fit`` accepts only acquired labels. A seeded internal split selects early
    stopping and tuning hyperparameters; target scaling uses its training rows
    exclusively. Each fit starts from a fresh seeded initialization. At least two
    labeled rows are required; a one-row or constant training target uses unit
    scale. ``device`` must already be resolved by the resource configuration.
    Tuning uses a seeded Optuna RandomSampler or GridSampler with one job.
    ``batch_size`` is the training batch; validation and prediction stream in
    batches of ``inference_batch_size`` (default ``batch_size``), which also
    bounds the memory used when ``X`` is a lazy :class:`PackedFingerprints` view.
    """

    CHECKPOINT_VERSION = 1
    _PARAMETERS = (
        "n_blocks",
        "d_block",
        "k",
        "dropout",
        "learning_rate",
        "weight_decay",
        "epochs",
        "batch_size",
        "patience",
        "validation_fraction",
        "seed",
        "device",
        "auto_tune",
        "tuning_method",
        "n_trials",
        "inference_batch_size",
    )

    def __init__(
        self,
        *,
        n_blocks: int = 3,
        d_block: int = 128,
        k: int = 16,
        dropout: float = 0.1,
        learning_rate: float = 0.002,
        weight_decay: float = 0.0003,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10,
        validation_fraction: float = 0.2,
        seed: int = 42,
        device: str = "cpu",
        auto_tune: bool = False,
        tuning_method: str = "random",
        n_trials: int = 10,
        inference_batch_size: int | None = None,
    ) -> None:
        parameters = locals().copy()
        for name in self._PARAMETERS:
            setattr(self, name, parameters[name])
        self._validate_parameters()
        self.model_type = "tabm"
        self.model: TabM | None = None
        self.is_fitted = False
        self.best_params_: dict[str, Any] | None = None
        self.tuning_trials_: list[dict[str, Any]] = []
        self.history_: list[dict[str, Any]] = []

    def _validate_parameters(self) -> None:
        for name in ("n_blocks", "d_block", "k", "epochs", "batch_size", "patience", "n_trials"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.inference_batch_size is not None and (
            type(self.inference_batch_size) is not int or self.inference_batch_size < 1
        ):
            raise ValueError("inference_batch_size must be None or a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("dropout", "learning_rate", "weight_decay", "validation_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if type(self.auto_tune) is not bool:
            raise ValueError("auto_tune must be a boolean")
        if self.tuning_method not in ("random", "grid"):
            raise ValueError("tuning_method must be 'random' or 'grid'")
        if self.device not in ("cpu", "cuda:0", "cuda:1"):
            raise ValueError("device must be 'cpu', 'cuda:0', or 'cuda:1' after resource configuration")
        self._validate_device_mapping()

    def _validate_device_mapping(self) -> None:
        """Reject an inherited CUDA mask that could address a forbidden GPU."""
        if self.device == "cpu":
            return
        mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        if mask is not None:
            tokens = [token.strip() for token in mask.split(",") if token.strip()]
            if any(token not in {"0", "1"} for token in tokens) or len(tokens) != len(set(tokens)):
                raise ValueError("CUDA_VISIBLE_DEVICES must contain unique numeric physical GPU IDs 0 or 1")
            logical_index = int(self.device.split(":")[1])
            if logical_index >= len(tokens):
                raise ValueError(f"Requested logical {self.device} is excluded by CUDA_VISIBLE_DEVICES")

    def get_params(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._PARAMETERS}

    def _make_model(self, params: dict[str, Any]) -> TabM:
        return TabM.make(
            n_num_features=self.n_features_in_,
            d_out=1,
            n_blocks=params["n_blocks"],
            d_block=params["d_block"],
            k=params["k"],
            dropout=params["dropout"],
        ).to(self.device)

    def fit(self, X: Any, y: Any, groups: Any = None) -> TabMRegressor:
        """Fit acquired labels, optionally grouping validation by molecular identity.

        ``groups`` supplies one string group identifier per row. Molecular
        callers should pass canonical isomeric SMILES so repeated structures
        cannot occur in both the training and internal validation partition.
        """
        X = _features(X)
        y = _targets(y, len(X))
        if len(X) < 2:
            raise ValueError("At least two acquired labeled samples are needed for training and validation")
        self._validate_device_mapping()
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {self.device}, but CUDA is unavailable")
        self.is_fitted = False
        self.n_features_in_, self.n_samples_fit_ = X.shape[1], len(X)
        n_validation = max(1, min(len(X) - 1, math.ceil(len(X) * self.validation_fraction)))
        self.grouped_validation_ = groups is not None
        if groups is None:
            order = np.random.default_rng(self.seed).permutation(len(X))
            self.validation_indices_, self.train_indices_ = order[:n_validation], order[n_validation:]
        else:
            group_array = np.asarray(groups, dtype=object)
            if group_array.shape != (len(X),) or any(
                not isinstance(group, str) or not group for group in group_array
            ):
                raise ValueError("groups must contain one nonempty string identifier per row")
            grouped_rows: dict[str, list[int]] = {}
            for index, group in enumerate(group_array):
                grouped_rows.setdefault(group, []).append(index)
            if len(grouped_rows) < 2:
                raise ValueError("Internal validation requires at least two distinct acquired groups")
            keys = sorted(grouped_rows)
            order = np.random.default_rng(self.seed).permutation(len(keys))
            validation = []
            for group_index in order[:-1]:
                validation.extend(grouped_rows[keys[group_index]])
                if len(validation) >= n_validation:
                    break
            self.validation_indices_ = np.sort(np.asarray(validation, dtype=np.int64))
            self.train_indices_ = np.setdiff1d(np.arange(len(X)), self.validation_indices_)
        training_y = y[self.train_indices_]
        self.target_mean_, self.target_scale_ = float(training_y.mean()), float(training_y.std(ddof=0))
        if self.target_scale_ <= np.finfo(np.float64).eps:
            self.target_scale_ = 1.0
        standardized_y = ((y - self.target_mean_) / self.target_scale_).astype(np.float32)
        if not np.isfinite(standardized_y).all():
            raise ValueError("Target scaling produced non-finite values; check target magnitude")
        self.best_params_, self.tuning_trials_ = None, []
        self.effective_params_ = self.get_params()
        if self.auto_tune:
            self.best_params_ = self._tune(X, standardized_y)
            self.effective_params_.update(self.best_params_)
        self._train(X, standardized_y, self.effective_params_)
        self.is_fitted = True
        return self

    def _train(self, X: np.ndarray, y: np.ndarray, params: dict[str, Any]) -> None:
        cuda_devices = [int(self.device.split(":")[1])] if self.device.startswith("cuda") else []
        # Isolate initialization and dropout RNG without initializing unused GPUs.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(self.seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(self.seed)
            self.model = self._make_model(params)
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
            )
            train_x, train_y = (
                torch.from_numpy(X[self.train_indices_]),
                torch.from_numpy(y[self.train_indices_]),
            )
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            self.history_, self.best_validation_loss_ = [], math.inf
            best_state, stale_epochs = None, 0
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                permutation = torch.randperm(len(train_x), generator=generator)
                train_loss = 0.0
                for indices in permutation.split(self.batch_size):
                    batch_x = train_x[indices].to(self.device, torch.float32)
                    batch_y = train_y[indices].to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    members = self.model(batch_x).squeeze(-1)
                    # Mean(member losses), NOT loss(mean prediction).
                    loss = (members - batch_y[:, None]).square().mean()
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Non-finite TabM training loss; check data and learning_rate")
                    loss.backward()
                    optimizer.step()
                    train_loss += float(loss.detach()) * len(indices)
                validation_loss = self._validation_mse(
                    X[self.validation_indices_], y[self.validation_indices_]
                )
                if not math.isfinite(validation_loss):
                    raise RuntimeError("Non-finite TabM validation loss")
                self.history_.append(
                    {
                        "epoch": epoch,
                        "train_member_mse_standardized": train_loss / len(train_x),
                        "validation_ensemble_mse_standardized": validation_loss,
                    }
                )
                if validation_loss < self.best_validation_loss_:
                    self.best_validation_loss_, self.best_epoch_ = validation_loss, epoch
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in self.model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= self.patience:
                        break
            if best_state is None:
                raise RuntimeError("Training did not produce a valid checkpoint")
            self.model.load_state_dict(best_state)
            self.model.eval()
            self.n_epochs_ = len(self.history_)

    @torch.inference_mode()
    def _validation_mse(self, X: np.ndarray, y: np.ndarray) -> float:
        assert self.model is not None
        self.model.eval()
        sum_squared_errors = 0.0
        batch = self._inference_batch()
        for start in range(0, len(X), batch):
            stop = start + batch
            mean = self.model(_batch(X[start:stop], self.device)).squeeze(-1).mean(dim=1).double()
            target = torch.from_numpy(y[start:stop]).to(self.device).double()
            sum_squared_errors += float((mean - target).square().sum())
        return sum_squared_errors / len(X)

    def _tune(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("auto_tune requires Optuna; install the project's tuning extra") from exc
        search_space = {
            "n_blocks": [1, 2, 3],
            "d_block": [64, 128, 256],
            "dropout": [0.0, 0.1, 0.2],
            "learning_rate": [0.0005, 0.002, 0.005],
            "weight_decay": [0.0, 0.0003],
        }
        sampler = (
            optuna.samplers.GridSampler(search_space, seed=self.seed)
            if self.tuning_method == "grid"
            else optuna.samplers.RandomSampler(seed=self.seed)
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial: Any) -> float:
            sampled = {
                name: trial.suggest_categorical(name, choices) for name, choices in search_space.items()
            }
            parameters = self.get_params()
            parameters.update(sampled)
            self._train(X, y, parameters)
            self.tuning_trials_.append(
                {
                    "number": trial.number,
                    "params": sampled,
                    "validation_ensemble_mse_standardized": self.best_validation_loss_,
                    "best_epoch": self.best_epoch_,
                }
            )
            return self.best_validation_loss_

        grid_size = math.prod(map(len, search_space.values()))
        limit = min(self.n_trials, grid_size) if self.tuning_method == "grid" else self.n_trials
        study.optimize(objective, n_trials=limit, n_jobs=1, show_progress_bar=False)
        return dict(study.best_params)

    def _inference_batch(self) -> int:
        return self.inference_batch_size or self.batch_size

    def _require_fitted(self) -> None:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("TabMRegressor is not fitted; call fit or load first")

    @torch.inference_mode()
    def predict_members(self, X: Any) -> np.ndarray:
        """Return member predictions in target units with shape ``(n, k)``."""
        self._require_fitted()
        X = _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features; expected {self.n_features_in_}")
        assert self.model is not None
        self.model.eval()
        output = np.empty((len(X), self.effective_params_["k"]), dtype=np.float64)
        batch = self._inference_batch()
        for start in range(0, len(X), batch):
            stop = start + batch
            output[start:stop] = self.model(_batch(X[start:stop], self.device)).squeeze(-1).cpu().numpy()
        output *= self.target_scale_
        output += self.target_mean_
        if not np.isfinite(output).all():
            raise RuntimeError("TabM produced non-finite predictions")
        return output

    def predict(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return mean and population standard deviation in target units."""
        self._require_fitted()
        X = _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features; expected {self.n_features_in_}")
        mean, std = np.empty(len(X)), np.empty(len(X))
        batch = self._inference_batch()
        for start in range(0, len(X), batch):
            stop = start + batch
            members = self.predict_members(X[start:stop])
            mean[start:stop], std[start:stop] = members.mean(axis=1), members.std(axis=1, ddof=0)
        return mean, std

    def evaluate_all_metrics(self, X: Any, y: Any) -> dict[str, Any]:
        """Return common regression metrics, with undefined R²/Spearman as None."""
        from .evaluation import regression_metrics

        mean, _ = self.predict(X)
        target = _targets(y, len(mean))
        if len(target) == 0:
            raise ValueError("Evaluation requires at least one row")
        return {**regression_metrics(target, mean), "n_samples": len(target)}

    def score(self, X: Any, y: Any) -> float | None:
        """Return R²; constant targets or one-row evaluation return None."""
        return self.evaluate_all_metrics(X, y)["r2"]

    def get_model_info(self) -> dict[str, Any]:
        info = {
            "model_type": "tabm",
            "implementation": "tabm.TabM",
            "upstream_repository": "https://github.com/yandex-research/tabm",
            "tabm_version": version("tabm"),
            "torch_version": str(torch.__version__),
            "is_fitted": self.is_fitted,
            "parameters": self.get_params(),
            "best_params": self.best_params_,
            "uncertainty": "ensemble_population_standard_deviation",
        }
        if self.is_fitted:
            info.update(
                {
                    "effective_parameters": self.effective_params_,
                    "n_features": self.n_features_in_,
                    "n_samples_fit": self.n_samples_fit_,
                    "n_training": len(self.train_indices_),
                    "n_validation": len(self.validation_indices_),
                    "grouped_validation": self.grouped_validation_,
                    "target_mean": self.target_mean_,
                    "target_scale": self.target_scale_,
                    "best_epoch": self.best_epoch_,
                    "epochs_run": self.n_epochs_,
                    "best_validation_mse_standardized": self.best_validation_loss_,
                }
            )
        return info

    def save(self, directory: str | Path) -> Path:
        """Save weights-only tensors plus JSON architecture and provenance."""
        self._require_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        weights_path = directory / "tabm_weights.pt"
        assert self.model is not None
        torch.save(
            {name: tensor.detach().cpu() for name, tensor in self.model.state_dict().items()}, weights_path
        )
        metadata = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            "model_info": self.get_model_info(),
            "train_indices": self.train_indices_.tolist(),
            "validation_indices": self.validation_indices_.tolist(),
            "history": self.history_,
            "tuning_trials": self.tuning_trials_,
        }
        (directory / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path, *, device: str = "cpu") -> TabMRegressor:
        """Load with ``weights_only=True``; never unpickle a saved model."""
        directory = Path(directory)
        metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("checkpoint_version") != cls.CHECKPOINT_VERSION:
            raise ValueError(
                "Unsupported checkpoint version; legacy surrogate MLP checkpoints are incompatible"
            )
        weights_path = directory / "tabm_weights.pt"
        if hashlib.sha256(weights_path.read_bytes()).hexdigest() != metadata["weights_sha256"]:
            raise ValueError("Checkpoint SHA256 mismatch")
        info = metadata["model_info"]
        params = dict(info["parameters"], device=device)
        estimator = cls(**params)
        estimator.n_features_in_, estimator.n_samples_fit_ = info["n_features"], info["n_samples_fit"]
        estimator.effective_params_ = dict(info["effective_parameters"], device=device)
        estimator.target_mean_, estimator.target_scale_ = info["target_mean"], info["target_scale"]
        estimator.best_params_ = info["best_params"]
        estimator.best_epoch_, estimator.n_epochs_ = info["best_epoch"], info["epochs_run"]
        estimator.best_validation_loss_ = info["best_validation_mse_standardized"]
        estimator.grouped_validation_ = info["grouped_validation"]
        estimator.train_indices_ = np.asarray(metadata["train_indices"], dtype=np.int64)
        estimator.validation_indices_ = np.asarray(metadata["validation_indices"], dtype=np.int64)
        estimator.history_, estimator.tuning_trials_ = metadata["history"], metadata["tuning_trials"]
        estimator.model = estimator._make_model(estimator.effective_params_)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        estimator.model.load_state_dict(state, strict=True)
        estimator.model.eval()
        estimator.is_fitted = True
        return estimator

    def save_model(self, save_path: str | Path) -> Path:
        """Compatibility spelling for :meth:`save`."""
        return self.save(save_path)

    def load_model(self, load_path: str | Path) -> TabMRegressor:
        """Restore in place for compatibility with the legacy API."""
        restored = type(self).load(load_path, device=self.device)
        self.__dict__.update(restored.__dict__)
        return self
