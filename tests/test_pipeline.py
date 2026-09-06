"""Real official-TabM integration tests in isolated, resource-limited processes."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from active_learning_docking.provenance import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def tiny_mapping(output, data_path=None):
    return {
        "seed": 19,
        "output_dir": str(output),
        "resources": {"cpu_threads": 1, "device": "cpu", "gpu_ids": [0, 1]},
        "data": {"path": str(data_path or ROOT / "examples/replay.csv"), "score_column": "synthetic_score", "fingerprint_bits": 128},
        "model": {"n_blocks": 1, "d_block": 16, "k": 4, "epochs": 2, "batch_size": 8, "patience": 2},
        "learning": {"initial_size": 6, "batch_size": 4, "cycles": 4, "budget": 13},
        "evaluation": {"holdout_fraction": 0.25, "top_k": 4},
    }


def run_cli(mapping, config_path, expected=0):
    config_path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    completed = subprocess.run(
        [sys.executable, "-m", "active_learning_docking", "run", "--config", str(config_path)],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == expected, completed.stdout + completed.stderr
    return completed


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("pipeline")
    output = directory / "run"
    mapping = tiny_mapping(output)
    run_cli(mapping, directory / "config.yaml")
    return directory, output, mapping


def test_budget_includes_initial_queries_and_caps_last_batch(completed_run):
    _, output, _ = completed_run
    observations = pd.read_csv(output / "observations.csv")
    summary = json.loads((output / "summary.json").read_text())
    assert len(observations) == 13
    assert observations.groupby("cycle").size().tolist() == [6, 4, 3]
    assert observations["attempt"].tolist() == list(range(1, 14))
    assert observations.compound_id.is_unique
    assert summary["stop_reason"] == "budget"
    assert summary["cycles_completed"] == 2
    assert summary["successful"] == summary["attempted"] == 13


def test_holdout_absent_from_queries_internal_splits_and_acquisition(completed_run):
    _, output, _ = completed_run
    partition = pd.read_csv(output / "split.csv")
    holdout = set(partition.loc[partition.partition == "holdout", "compound_id"])
    pool = set(partition.loc[partition.partition == "pool", "compound_id"])
    assert len(holdout) == 12 and len(pool) == 36
    assert not holdout & set(pd.read_csv(output / "observations.csv").compound_id)
    for cycle in sorted(output.glob("cycle_*")):
        split = pd.read_csv(cycle / "training_split.csv")
        assert set(split.compound_id) <= pool
        assert {"train", "validation"} == set(split.partition)
        path = cycle / "pool_predictions_before_acquisition.csv"
        if path.exists():
            assert set(pd.read_csv(path).compound_id) <= pool
        assert set(pd.read_csv(cycle / "holdout_predictions.csv").compound_id) == holdout


def test_checkpoint_reproduces_exported_predictions(completed_run):
    from active_learning_docking.data import featurize
    from active_learning_docking.model import TabMRegressor

    _, output, _ = completed_run
    exported = pd.read_csv(output / "predictions.csv", float_precision="round_trip")
    restored = TabMRegressor.load(output / "model")
    mean, std = restored.predict(featurize(exported.smiles, n_bits=128))
    np.testing.assert_array_equal(mean, exported.prediction.to_numpy())
    np.testing.assert_array_equal(std, exported.uncertainty.to_numpy())
    metadata = json.loads((output / "model/model_metadata.json").read_text())
    assert metadata["model_info"]["implementation"] == "tabm.TabM"
    assert metadata["model_info"]["n_samples_fit"] == 13


def test_manifest_captures_actual_inputs_artifact_integrity_and_resources(completed_run):
    _, output, _ = completed_run
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["input_sha256"][str(ROOT / "examples/replay.csv")] == sha256_file(ROOT / "examples/replay.csv")
    assert manifest["resources"]["cpu_threads"] == 1
    assert len(manifest["resources"]["cpu_affinity"]) == 1
    assert manifest["resources"]["cuda_visible_devices"] == ""
    assert manifest["resources"]["deterministic_algorithms"]
    assert "tabm" in {name.lower() for name in manifest["distributions"]}
    for name, checksum in manifest["artifact_sha256"].items():
        assert sha256_file(output / name) == checksum


def test_changing_only_holdout_labels_cannot_change_training_or_selection(completed_run):
    directory, first, mapping = completed_run
    partition = pd.read_csv(first / "split.csv")
    heldout_ids = set(partition.loc[partition.partition == "holdout", "compound_id"])
    frame = pd.read_csv(ROOT / "examples/replay.csv", float_precision="round_trip")
    frame.loc[frame["Catalog ID"].isin(heldout_ids), "synthetic_score"] -= 1000
    changed_data = directory / "changed_holdout.csv"
    frame.to_csv(changed_data, index=False)
    second = directory / "changed_run"
    changed = tiny_mapping(second, changed_data)
    run_cli(changed, directory / "changed.yaml")
    for name in ("observations.csv", "predictions.csv", "training_history.csv", "split.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    first_metrics, second_metrics = [pd.read_csv(path / "metrics.csv") for path in (first, second)]
    assert not first_metrics.holdout_rmse.equals(second_metrics.holdout_rmse)
    assert first_metrics.discovery_top_k_recall.equals(second_metrics.discovery_top_k_recall)


def test_failure_is_nonzero_and_recorded_with_row_audit(tmp_path):
    data = tmp_path / "invalid.csv"
    data.write_text("Catalog ID,smiles,synthetic_score\nbad,not-a-smiles,-1\n")
    output = tmp_path / "failed"
    mapping = tiny_mapping(output, data)
    mapping["data"]["invalid_policy"] = "skip"
    completed = run_cli(mapping, tmp_path / "invalid.yaml", expected=1)
    assert "No valid compounds" in completed.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "LibraryValidationError" in manifest["error"]
    assert not (output / "summary.json").exists()
    rejected = pd.read_csv(output / "rejected_rows.csv")
    assert rejected.reason.tolist() == ["invalid_smiles"]


def test_zero_cycles_runs_initial_batch_only_and_does_not_overwrite(tmp_path):
    output = tmp_path / "initial_only"
    mapping = tiny_mapping(output)
    mapping["learning"]["cycles"] = 0
    run_cli(mapping, tmp_path / "initial.yaml")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["attempted"] == 6
    assert summary["cycles_completed"] == 0
    assert summary["stop_reason"] == "cycles"
    checksum = sha256_file(output / "summary.json")
    run_cli(mapping, tmp_path / "initial.yaml", expected=1)
    assert sha256_file(output / "summary.json") == checksum


def test_exhausting_pool_stops_without_requery_or_budget_padding(tmp_path):
    data = tmp_path / "small.csv"
    pd.read_csv(ROOT / "examples/replay.csv").iloc[:8].to_csv(data, index=False)
    output = tmp_path / "exhausted"
    mapping = tiny_mapping(output, data)
    run_cli(mapping, tmp_path / "exhausted.yaml")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["stop_reason"] == "pool_exhausted"
    assert summary["attempted"] == summary["pool_size"] == 6
    assert pd.read_csv(output / "observations.csv").compound_id.is_unique


CONTROLLED_ORACLE_SCRIPT = """
import sys
import threading
import time
from unittest.mock import patch
from active_learning_docking.config import load_config
from active_learning_docking.pipeline import run_experiment

INSTANCES = []

class ControlledOracle:
    \"\"\"Fail one designated compound; score the others by SMILES length.\"\"\"

    def __init__(self, *args, **kwargs):
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()
        INSTANCES.append(self)

    def evaluate(self, compound_id, smiles, output_dir):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(0.05)
            if compound_id == sys.argv[2]:
                return {"status": "failed", "score": None, "error": "controlled failure", "pose": None}
            return {"status": "ok", "score": -float(len(smiles)), "error": None, "pose": None}
        finally:
            with self.lock:
                self.active -= 1

with patch("active_learning_docking.docking.DockingOracle", ControlledOracle):
    run_experiment(load_config(sys.argv[1]))
print("peak_concurrency", INSTANCES[0].peak)
"""


def initial_sample_ids(mapping):
    """Replicate the seeded cycle-0 sample so a queried compound can be forced to fail."""
    from active_learning_docking.data import load_library

    library, _ = load_library(mapping["data"]["path"], score_column=None)
    rng = np.random.default_rng(np.random.SeedSequence(mapping["seed"]).spawn(2)[0])
    positions = rng.choice(np.arange(len(library)), size=mapping["learning"]["initial_size"], replace=False)
    return library.compound_id.iloc[positions].astype(str).tolist()


def run_controlled_docking(tmp_path, name, workers):
    """Run the real pipeline with an in-process oracle; return (output, stdout)."""
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("Controlled oracle test; no docking engine reads this fixture.\n")
    output = tmp_path / name
    mapping = tiny_mapping(output)
    mapping.update(mode="docking", docking={"receptor": str(receptor), "center": [0, 0, 0], "workers": workers})
    mapping["resources"]["cpu_threads"] = 2
    mapping["evaluation"]["holdout_fraction"] = 0
    mapping["learning"]["budget"] = 9
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(mapping))
    failing_id = initial_sample_ids(mapping)[0]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    completed = subprocess.run([sys.executable, "-c", CONTROLLED_ORACLE_SCRIPT, str(path), failing_id],
                               cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return output, completed.stdout


def test_failed_docking_attempt_consumes_budget_and_never_becomes_training_label(tmp_path):
    """A controlled oracle failure exercises the real pipeline and TabM trainer."""
    output, _ = run_controlled_docking(tmp_path, "partial_failure", workers=1)
    summary = json.loads((output / "summary.json").read_text())
    observations = pd.read_csv(output / "observations.csv")
    assert summary["attempted"] == 9
    assert summary["successful"] == 8
    assert summary["stop_reason"] == "budget"
    assert observations.compound_id.is_unique
    failed_id = observations.loc[observations.status == "failed", "compound_id"].item()
    assert failed_id not in set(pd.read_csv(output / "final_results.csv").compound_id)
    for cycle in output.glob("cycle_*"):
        assert failed_id not in set(pd.read_csv(cycle / "training_split.csv").compound_id)


def test_parallel_docking_workers_do_not_change_any_scientific_record(tmp_path):
    from active_learning_docking.experiments import compare_runs

    sequential, _ = run_controlled_docking(tmp_path, "sequential", workers=1)
    parallel, stdout = run_controlled_docking(tmp_path, "parallel", workers=2)
    assert "peak_concurrency 2" in stdout
    report = compare_runs(sequential, parallel)
    assert report["passed"], report["checks"]
    assert (sequential / "observations.csv").read_bytes() == (parallel / "observations.csv").read_bytes()


def test_beta_schedule_is_recorded_per_cycle_and_changes_selection(tmp_path):
    constant = tmp_path / "constant"
    run_cli(tiny_mapping(constant), tmp_path / "constant.yaml")
    scheduled = tmp_path / "scheduled"
    mapping = tiny_mapping(scheduled)
    mapping["learning"].update(beta_schedule="linear", beta_final=0.0, budget=18)
    run_cli(mapping, tmp_path / "scheduled.yaml")
    metrics = pd.read_csv(scheduled / "metrics.csv")
    assert np.isnan(metrics.acquisition_beta.iloc[0])  # cycle 0 is the random initial sample
    assert metrics.acquisition_beta.tolist()[1:] == pytest.approx([1.0, 2 / 3, 1 / 3])
    assert pd.read_csv(constant / "metrics.csv").acquisition_beta.tolist()[1:] == [1.0, 1.0]
    first = pd.read_csv(constant / "cycle_001/queries.csv").compound_id.tolist()
    assert pd.read_csv(scheduled / "cycle_001/queries.csv").compound_id.tolist() == first
    for cycle, beta in ((2, 2 / 3), (3, 1 / 3)):
        pool = pd.read_csv(scheduled / f"cycle_{cycle:03d}/pool_predictions_before_acquisition.csv")
        expected = pool.assign(value=pool.prediction - beta * pool.uncertainty).sort_values(
            ["value", "compound_id"]).compound_id.head(4).tolist()
        assert pd.read_csv(scheduled / f"cycle_{cycle:03d}/queries.csv").compound_id.tolist() == expected
    resolved = yaml.safe_load((scheduled / "config.resolved.yaml").read_text())
    assert resolved["learning"]["beta_schedule"] == "linear"


def test_compatibility_entry_points_wrap_the_package_api():
    import Activedocking
    import main
    import unified_model
    from active_learning_docking.model import TabMRegressor

    assert callable(main.main)
    assert issubclass(unified_model.UnifiedModel, TabMRegressor)
    assert unified_model.UnifiedModel(n_blocks=1, d_block=8, k=2, epochs=1).model_type == "tabm"
    with pytest.raises(TypeError):
        Activedocking.ActiveLearningWithConformers({"config": "dict"})
