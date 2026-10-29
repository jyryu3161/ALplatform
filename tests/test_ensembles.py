"""Real local backends, weighted uncertainty, leakage isolation and checkpoint restore."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from active_learning_docking.config import EnsembleConfig, ModelConfig
from active_learning_docking.data import featurize
from active_learning_docking.model import TabMRegressor
from active_learning_docking.regressors import (
    ForestRegressor, RegressionEnsemble, create_regressor, load_regressor,
)
from active_learning_docking.store import PackedFingerprints
from tests.test_pipeline import ROOT, run_cli, tiny_mapping


def small_model(models, strategy="mean", weights=None):
    # TabM-only values are rejected unless TabM is selected.
    tabm = dict(n_blocks=1, d_block=8, k=4, epochs=2) if {"tabm", "tamb"} & set(models) else {}
    return ModelConfig(
        models=models, ensemble=EnsembleConfig(strategy=strategy, weights=weights),
        batch_size=8, inference_batch_size=7, **tabm,
        params={name: {"n_estimators": 5, "max_depth": 3}
                for name in models if name in {"random_forest", "extra_trees", "rf", "et"}},
    )


@pytest.fixture
def regression_data():
    X = np.random.default_rng(2).integers(0, 2, size=(32, 16), dtype=np.uint8)
    y = -2.0 * X[:, 0] + X[:, 1] - 5.0
    return X, y


@pytest.mark.parametrize("name,expected", [("tamb", TabMRegressor), ("rf", ForestRegressor), ("et", ForestRegressor)])
def test_single_selected_model_alias_has_no_outer_ensemble_and_restores(tmp_path, regression_data, name, expected):
    X, y = regression_data
    model = create_regressor(small_model([name]), seed=3).fit(PackedFingerprints.from_dense(X), y)
    assert isinstance(model, expected)
    assert not isinstance(model, RegressionEnsemble)
    model.save(tmp_path / "model")
    restored = load_regressor(tmp_path / "model")
    for actual, reference in zip(restored.predict(X), model.predict(X)):
        np.testing.assert_array_equal(actual, reference)
    assert all(values.shape == (0,) for values in restored.predict(X[:0]))


@pytest.mark.parametrize("strategy,weights", [("mean", None), ("weighted", [1.0, 3.0])])
def test_real_tabm_forest_ensemble_math_lazy_batches_and_restore(tmp_path, regression_data, strategy, weights):
    X, y = regression_data
    packed = PackedFingerprints.from_dense(X)
    config = small_model(["tabm", "random_forest"], strategy, weights)
    model = create_regressor(config, seed=8).fit(packed, y)
    assert model.members_["tabm"].k == 4
    assert len(model.members_["random_forest"].estimator.estimators_) == 5
    expected_weights = np.array([0.5, 0.5] if weights is None else [0.25, 0.75])
    np.testing.assert_array_equal(model.weights_, expected_weights)
    member_predictions = [member.predict(X) for member in model.members_.values()]
    means = np.array([prediction[0] for prediction in member_predictions])
    deviations = np.array([prediction[1] for prediction in member_predictions])
    expected_mean = np.sum(expected_weights[:, None] * means, axis=0)
    expected_std = np.sqrt(np.sum(expected_weights[:, None] * (
        deviations ** 2 + (means - expected_mean) ** 2), axis=0))
    actual = model.predict(packed)
    np.testing.assert_allclose(actual[0], expected_mean, rtol=0, atol=1e-6)
    np.testing.assert_allclose(actual[1], expected_std, rtol=0, atol=1e-6)
    # A lazy reordered view exercises partial inference batches and repeated rows.
    selected = np.array([31, 2, 2, 0, 12, 4, 8, 15, 19])
    for dense, lazy in zip(model.predict(X[selected]), model.predict(packed.take(selected))):
        np.testing.assert_array_equal(dense, lazy)
    assert all(value.shape == (0,) for value in model.predict(packed.take(np.empty(0, dtype=int))))
    model.save(tmp_path / "ensemble")
    restored = load_regressor(tmp_path / "ensemble")
    assert restored.get_model_info() == model.get_model_info()
    for before, after in zip(actual, restored.predict(packed)):
        np.testing.assert_array_equal(before, after)


def test_validation_weights_use_disjoint_group_probes_then_refit_all_acquired_rows(monkeypatch, tmp_path):
    X = np.arange(48, dtype=np.float32).reshape(24, 2)
    y = -X[:, 0] / 5 + np.sin(X[:, 1])
    groups = np.repeat([f"molecule_{i}" for i in range(12)], 2)
    calls = []
    original_fit = ForestRegressor.fit

    def record_fit(self, features, targets, groups=None):
        calls.append((self.model_type, np.asarray(features).copy(), np.asarray(targets).copy(), groups.copy()))
        return original_fit(self, features, targets, groups=groups)

    monkeypatch.setattr(ForestRegressor, "fit", record_fit)
    config = small_model(["random_forest", "extra_trees"], "validation")
    model = create_regressor(config, seed=23).fit(X, y, groups=groups)
    train, valid = model.weight_train_indices_, model.weight_validation_indices_
    assert len(valid) and not set(groups[train]) & set(groups[valid])
    assert len(calls) == 4
    for _, features, targets, seen_groups in calls[:2]:
        np.testing.assert_array_equal(features, X[train])
        np.testing.assert_array_equal(targets, y[train])
        assert not set(seen_groups) & set(groups[valid])
    for _, features, targets, _ in calls[2:]:
        np.testing.assert_array_equal(features, X)
        np.testing.assert_array_equal(targets, y)
    errors = np.array(list(model.validation_mse_.values()))
    inverse = 1 / np.maximum(errors, 1e-12)
    np.testing.assert_allclose(model.weights_, inverse / inverse.sum())
    assert all(member.n_samples_fit_ == len(X) for member in model.members_.values())
    # Forest restore refits from its recorded arrays without groups.
    monkeypatch.setattr(ForestRegressor, "fit", original_fit)
    model.save(tmp_path / "validation")
    restored = load_regressor(tmp_path / "validation")
    np.testing.assert_array_equal(restored.weight_validation_indices_, valid)
    np.testing.assert_array_equal(restored.weights_, model.weights_)
    for before, after in zip(model.predict(X), restored.predict(X)):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("models,groups", [
    (["tabm", "random_forest"], ["a", "a", "b", "b"]),
    (["nori", "random_forest"], ["a", "a", "b", "b"]),
    (["random_forest", "extra_trees"], ["a", "a", "a", "a"]),
])
def test_validation_weighting_rejects_insufficient_independent_acquired_groups(monkeypatch, models, groups):
    if "nori" in models:
        # The split must fail before any optional backend is fitted or downloaded.
        monkeypatch.setattr("active_learning_docking.regressors._make_member", lambda *args: None)
    model = create_regressor(small_model(models, "validation"))
    with pytest.raises(ValueError, match="distinct acquired groups"):
        model.fit(np.ones((4, 2), dtype=np.float32), [-3, -4, -3, -4], groups=groups)


@pytest.fixture(scope="module")
def ensemble_replay(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ensemble_replay")
    output = directory / "run"
    mapping = tiny_mapping(output)
    mapping["learning"].update(initial_size=8, batch_size=3, cycles=1, budget=11)
    mapping["model"].update(models=["tamb", "random_forest"],
                            ensemble={"strategy": "validation"}, inference_batch_size=7,
                            params={"random_forest": {"n_estimators": 5, "max_depth": 3}})
    run_cli(mapping, directory / "ensemble.yaml")
    return directory, output, mapping


def test_cli_ensemble_replay_records_weights_and_restores_exported_predictions(ensemble_replay):
    _, output, _ = ensemble_replay
    summary = json.loads((output / "summary.json").read_text())
    assert summary["models"] == ["tabm", "random_forest"]
    assert summary["attempted"] == summary["successful"] == 11
    weights = [summary["final_metrics"][f"ensemble_weight_{name}"] for name in summary["models"]]
    assert sum(weights) == pytest.approx(1)
    assert all(weight > 0 for weight in weights)
    exported = pd.read_csv(output / "predictions.csv", float_precision="round_trip")
    restored = load_regressor(output / "model")
    mean, std = restored.predict(featurize(exported.smiles, n_bits=128))
    np.testing.assert_array_equal(mean, exported.prediction.to_numpy())
    np.testing.assert_array_equal(std, exported.uncertainty.to_numpy())
    partition = pd.read_csv(output / "split.csv")
    holdout = set(partition.loc[partition.partition == "holdout", "compound_id"])
    acquired = set(pd.read_csv(output / "observations.csv").compound_id)
    for cycle in output.glob("cycle_*"):
        split = pd.read_csv(cycle / "ensemble_validation_split.csv")
        assert set(split.compound_id) <= acquired
        assert not set(split.compound_id) & holdout
        assert set(split.partition) == {"weight_train", "weight_validation"}
        members = pd.read_csv(cycle / "member_training_splits.csv")
        assert set(members.model) == {"tabm", "random_forest"}
        assert set(members.compound_id) == set(split.compound_id)
        assert set(members.loc[members.model == "tabm", "partition"]) == {"train", "validation"}
        assert set(members.loc[members.model == "random_forest", "partition"]) == {"train"}


def test_changing_holdout_labels_cannot_change_ensemble_weights_models_or_acquisition(ensemble_replay):
    directory, first, mapping = ensemble_replay
    partitions = pd.read_csv(first / "split.csv")
    heldout = set(partitions.loc[partitions.partition == "holdout", "compound_id"])
    data = pd.read_csv(ROOT / "examples/replay.csv", float_precision="round_trip")
    data.loc[data["Catalog ID"].isin(heldout), "synthetic_score"] -= 1000
    source = directory / "changed_holdout.csv"
    data.to_csv(source, index=False)
    second = directory / "changed_run"
    changed = deepcopy(mapping)
    changed["output_dir"], changed["data"]["path"] = str(second), str(source)
    run_cli(changed, directory / "changed.yaml")
    for name in ("observations.csv", "predictions.csv", "training_history.csv", "split.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    for cycle in first.glob("cycle_*"):
        relative = cycle.relative_to(first)
        for name in ("ensemble_validation_split.csv", "member_training_splits.csv", "model/model_metadata.json"):
            assert (cycle / name).read_bytes() == (second / relative / name).read_bytes(), name
    before, after = [pd.read_csv(path / "metrics.csv") for path in (first, second)]
    assert not before.holdout_rmse.equals(after.holdout_rmse)
    assert before.ensemble_weight_tabm.equals(after.ensemble_weight_tabm)
    assert before.ensemble_weight_random_forest.equals(after.ensemble_weight_random_forest)
