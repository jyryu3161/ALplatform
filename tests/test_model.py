"""Behavioral checks against the official TabM implementation."""

import json

import numpy as np
import pytest
import torch
from tabm import TabM

from active_learning_docking.model import TabMRegressor


@pytest.fixture
def data():
    rng = np.random.default_rng(102)
    X = rng.normal(size=(48, 6)).astype(np.float32)
    y = 1.7 * X[:, 0] - X[:, 1] + 0.3 * X[:, 2] ** 2 - 6
    return X, y


def small_model(**kwargs):
    parameters = dict(n_blocks=1, d_block=16, k=4, epochs=5, batch_size=7, patience=3, seed=17, device="cpu")
    parameters.update(kwargs)
    return TabMRegressor(**parameters)


def test_official_architecture_and_prediction_semantics(data):
    X, y = data
    estimator = small_model().fit(X, y)
    assert isinstance(estimator.model, TabM)
    assert estimator.model(torch.from_numpy(X[:3])).shape == (3, 4, 1)
    members = estimator.predict_members(X)
    mean, std = estimator.predict(X)
    assert members.shape == (48, 4)
    np.testing.assert_array_equal(mean, members.mean(axis=1))
    np.testing.assert_array_equal(std, members.std(axis=1, ddof=0))
    assert np.all(std > 0)
    assert not estimator.model.training
    repeated = estimator.predict(X)
    np.testing.assert_array_equal(mean, repeated[0])
    np.testing.assert_array_equal(std, repeated[1])


def test_training_optimizes_member_losses_not_loss_of_mean(data):
    X, y = data
    estimator = small_model(epochs=1, batch_size=128, dropout=0).fit(X, y)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(estimator.seed)
        reference = TabM.make(n_num_features=6, d_out=1, n_blocks=1, d_block=16, k=4, dropout=0)
    train_x = torch.from_numpy(X[estimator.train_indices_])
    targets = torch.from_numpy(
        ((y.astype(np.float64) - estimator.target_mean_) / estimator.target_scale_).astype(np.float32)
    )[estimator.train_indices_]
    with torch.no_grad():
        predictions = reference(train_x).squeeze(-1)
        mean_member_loss = float((predictions - targets[:, None]).square().mean())
        loss_of_mean = float((predictions.mean(dim=1) - targets).square().mean())
    observed = estimator.history_[0]["train_member_mse_standardized"]
    assert observed == pytest.approx(mean_member_loss, abs=1e-6)
    assert observed > loss_of_mean + 1e-5


def test_fit_is_seeded_and_preserves_callers_rng(data):
    X, y = data
    torch_state = torch.random.get_rng_state().clone()
    first = small_model().fit(X, y)
    assert torch.equal(torch_state, torch.random.get_rng_state())
    np.random.default_rng(992).normal(size=100)
    torch.rand(5)
    second = small_model().fit(X, y)
    assert first.history_ == second.history_
    np.testing.assert_array_equal(first.predict_members(X), second.predict_members(X))
    np.testing.assert_array_equal(first.train_indices_, second.train_indices_)
    different = small_model(seed=18).fit(X, y)
    assert not np.array_equal(first.predict(X)[0], different.predict(X)[0])


def test_validation_is_disjoint_and_scaling_uses_training_only(data):
    X, y = data
    indices = np.random.default_rng(17).permutation(len(X))
    validation_indices = indices[:10]
    y = y.copy()
    y[validation_indices] = 1000
    estimator = small_model().fit(X, y)
    assert set(estimator.train_indices_).isdisjoint(estimator.validation_indices_)
    assert set(estimator.train_indices_) | set(estimator.validation_indices_) == set(range(len(X)))
    np.testing.assert_array_equal(estimator.validation_indices_, validation_indices)
    assert estimator.target_mean_ == pytest.approx(y[estimator.train_indices_].astype(np.float64).mean())
    assert estimator.target_scale_ == pytest.approx(y[estimator.train_indices_].astype(np.float64).std())
    assert estimator.target_mean_ < 0


def test_best_validation_checkpoint_is_restored(data):
    X, y = data
    estimator = small_model(learning_rate=0.15, epochs=30, patience=4).fit(X, y)
    losses = [row["validation_ensemble_mse_standardized"] for row in estimator.history_]
    assert estimator.best_epoch_ == int(np.argmin(losses)) + 1
    assert estimator.n_epochs_ > estimator.best_epoch_
    assert estimator.n_epochs_ < 30
    standardized_y = ((y.astype(np.float64) - estimator.target_mean_) / estimator.target_scale_).astype(
        np.float32
    )
    restored_loss = estimator._validation_mse(
        X[estimator.validation_indices_], standardized_y[estimator.validation_indices_]
    )
    assert restored_loss == min(losses)


def test_checkpoint_roundtrip_uses_tensors_and_json(data, tmp_path, monkeypatch):
    X, y = data
    original = small_model().fit(X, y)
    original.save(tmp_path)
    metadata = json.loads((tmp_path / "model_metadata.json").read_text())
    assert metadata["model_info"]["implementation"] == "tabm.TabM"
    assert metadata["model_info"]["tabm_version"] == "0.0.3"
    original_load = torch.load
    arguments = []

    def record_load(*args, **kwargs):
        arguments.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", record_load)
    restored = TabMRegressor.load(tmp_path)
    assert arguments[0]["weights_only"] is True
    assert original.get_model_info() == restored.get_model_info()
    assert original.history_ == restored.history_
    np.testing.assert_array_equal(original.predict_members(X), restored.predict_members(X))
    weights = tmp_path / "tabm_weights.pt"
    weights.write_bytes(weights.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="SHA256"):
        TabMRegressor.load(tmp_path)


@pytest.mark.parametrize("method", ["random", "grid"])
def test_tuning_is_seeded_and_best_trial_is_applied(data, method):
    pytest.importorskip("optuna")
    X, y = data
    parameters = dict(auto_tune=True, tuning_method=method, n_trials=2, epochs=2, batch_size=128)
    first = small_model(**parameters).fit(X[:24], y[:24])
    second = small_model(**parameters).fit(X[:24], y[:24])
    assert first.tuning_trials_ == second.tuning_trials_
    assert len(first.tuning_trials_) == 2
    best = min(first.tuning_trials_, key=lambda trial: trial["validation_ensemble_mse_standardized"])
    assert first.best_params_ == best["params"]
    assert first.best_validation_loss_ == best["validation_ensemble_mse_standardized"]
    for name, value in first.best_params_.items():
        assert first.effective_params_[name] == value
    assert len(first.train_indices_) + len(first.validation_indices_) == 24
    np.testing.assert_array_equal(first.predict_members(X), second.predict_members(X))


def test_compact_integer_features_match_float_features_exactly(tmp_path):
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, size=(40, 24)).astype(np.uint8)
    y = bits[:, :4].sum(axis=1) - 3.0 + 0.1 * bits[:, 5]
    compact = small_model().fit(bits, y)
    expanded = small_model().fit(bits.astype(np.float32), y)
    assert compact.history_ == expanded.history_
    np.testing.assert_array_equal(compact.predict_members(bits), expanded.predict_members(bits.astype(np.float32)))
    np.testing.assert_array_equal(compact.predict_members(bits.astype(bool)), compact.predict_members(bits))
    compact.save(tmp_path)
    np.testing.assert_array_equal(TabMRegressor.load(tmp_path).predict_members(bits), compact.predict_members(bits))
    with pytest.raises(ValueError, match="rectangular"):
        small_model().fit([[1, 2], [3]], [1, 2])


def test_tiny_constant_data_single_member_and_empty_prediction():
    X = np.array([[0, 1], [1, 0]], dtype=np.float32)
    estimator = small_model(k=1, batch_size=1, epochs=2).fit(X, [-6, -6])
    assert estimator.target_scale_ == 1
    assert estimator.score(X, [-6, -6]) is None
    mean, std = estimator.predict(X)
    assert np.isfinite(mean).all()
    np.testing.assert_array_equal(std, np.zeros(2))
    assert estimator.predict(np.empty((0, 2)))[0].shape == (0,)
    assert estimator.predict_members(np.empty((0, 2))).shape == (0, 1)
    with pytest.raises(ValueError, match="at least one row"):
        estimator.score(np.empty((0, 2)), [])


@pytest.mark.parametrize(
    "X,y,message",
    [
        ([[1, 2]], [1], "At least two"),
        ([[1, 2], [2, 1]], [1], "one scalar"),
        ([[1, 2], [2, 1]], [[1, 2], [3, 4]], "one scalar"),
        ([[1, float("nan")], [2, 1]], [1, 2], "finite"),
        ([[1, 2], [2, 1]], [float("inf"), 2], "finite"),
        ([1, 2], [1, 2], "shape"),
        ([[], []], [1, 2], "shape"),
    ],
)
def test_invalid_training_data_raise(X, y, message):
    with pytest.raises(ValueError, match=message):
        small_model().fit(X, y)


def test_predict_before_fit_and_wrong_feature_count(data):
    X, y = data
    with pytest.raises(RuntimeError, match="not fitted"):
        small_model().predict(X)
    estimator = small_model().fit(X, y)
    with pytest.raises(ValueError, match="expected 6"):
        estimator.predict(np.ones((5, 7)))


@pytest.mark.parametrize(
    "parameters",
    [
        {"k": 0},
        {"batch_size": True},
        {"epochs": 1.5},
        {"dropout": 1},
        {"learning_rate": 0},
        {"weight_decay": -1},
        {"validation_fraction": 0},
        {"seed": -1},
        {"auto_tune": "yes"},
        {"tuning_method": "tpe"},
        {"device": "cuda:2"},
        {"patience": 0},
        {"dropout": float("nan")},
    ],
)
def test_invalid_parameters_raise(parameters):
    with pytest.raises(ValueError):
        small_model(**parameters)


def test_grouped_internal_validation_keeps_duplicate_molecules_together(data, tmp_path):
    X, y = data
    groups = np.array([f"structure_{index // 3}" for index in range(len(X))])
    first = small_model().fit(X, y, groups=groups)
    second = small_model().fit(X, y, groups=groups)
    train_groups = set(groups[first.train_indices_])
    validation_groups = set(groups[first.validation_indices_])
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups | validation_groups == set(groups)
    assert first.target_mean_ == pytest.approx(y[first.train_indices_].astype(np.float64).mean())
    np.testing.assert_array_equal(first.predict(X)[0], second.predict(X)[0])
    first.save(tmp_path)
    restored = TabMRegressor.load(tmp_path)
    assert restored.get_model_info()["grouped_validation"] is True
    np.testing.assert_array_equal(first.validation_indices_, restored.validation_indices_)
    with pytest.raises(ValueError, match="two distinct acquired groups"):
        small_model().fit(X, y, groups=["same"] * len(X))
    with pytest.raises(ValueError, match="one nonempty string"):
        small_model().fit(X, y, groups=["too_short"])


@pytest.mark.parametrize("mask", ["2", "2,0", "GPU-unknown", "0,0"])
def test_direct_estimator_rejects_unsafe_gpu_masks(monkeypatch, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(ValueError, match="physical GPU IDs 0 or 1"):
        small_model(device="cuda:0")


def test_direct_estimator_respects_safe_gpu_remapping_without_initializing_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert small_model(device="cuda:0").device == "cuda:0"
    with pytest.raises(ValueError, match="excluded"):
        small_model(device="cuda:1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="excluded"):
        small_model(device="cuda:0")


def test_official_model_learns_known_signal_on_unseen_rows():
    rng = np.random.default_rng(411)
    X = rng.normal(size=(160, 6)).astype(np.float32)
    y = 2 * X[:, 0] - X[:, 1] - 6
    estimator = small_model(
        d_block=32, dropout=0, epochs=100, batch_size=32, patience=15, learning_rate=0.01
    ).fit(X[:128], y[:128])
    metrics = estimator.evaluate_all_metrics(X[128:], y[128:])
    assert metrics["r2"] > 0.95
    assert metrics["rmse"] < 0.2
