"""Real official-TabM integration tests in isolated, resource-limited processes."""

import csv
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


def test_optional_model_preflight_failure_spends_no_docking_budget(monkeypatch, tmp_path):
    import logging

    from active_learning_docking.config import from_dict
    from active_learning_docking.foundation import FoundationRegressor
    from active_learning_docking.pipeline import ActiveLearningExperiment

    config = from_dict({"data": {"path": "unused.csv"}, "model": {"models": ["nori"]}})
    experiment = ActiveLearningExperiment(config, {"device": "cpu"}, tmp_path, logging.getLogger(__name__))

    def unavailable(self):
        raise ImportError("optional runtime unavailable")

    def should_not_start(*args):
        pytest.fail("The pool/oracle must not be used when model preflight fails")

    monkeypatch.setattr(FoundationRegressor, "preflight", unavailable)
    monkeypatch.setattr(experiment, "prepare_library", should_not_start)
    monkeypatch.setattr(experiment, "make_oracle", should_not_start)
    with pytest.raises(ImportError, match="optional runtime unavailable"):
        experiment.run()
    assert experiment.observations == []


def tiny_mapping(output, data_path=None):
    return {
        "seed": 19,
        "output_dir": str(output),
        "resources": {"cpu_threads": 1, "device": "cpu", "gpu_ids": [0, 1]},
        "data": {"path": str(data_path or ROOT / "examples/replay.csv"), "score_column": "synthetic_score", "fingerprint_bits": 128},
        "model": {"models": ["tabm"], "n_blocks": 1, "d_block": 16, "k": 4, "epochs": 2, "batch_size": 8,
                  "patience": 2},
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


LEGACY_OBSERVATION_COLUMNS = ["compound_id", "smiles", "library_index", "cycle", "attempt", "score", "status",
                              "error", "pose", "prediction_before_query", "uncertainty_before_query"]
RESCORING_COLUMNS = ["docking_energy", "cnn_score", "cnn_affinity"]


def fake_docking_mapping(tmp_path, name, behaviour=None):
    """A small docking run whose engine and GNINA are the fake executables of ``tests.test_docking``."""
    from tests.test_docking import write_fake_gnina

    data = tmp_path / "candidates.csv"
    if not data.exists():
        pd.read_csv(ROOT / "examples/replay.csv").iloc[:10].to_csv(data, index=False)
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("ATOM      1  C   ALA A   1       1.000   2.000   3.000  1.00  0.00     0.000 C\n")
    mapping = tiny_mapping(tmp_path / name, data)
    mapping.update(mode="docking")
    mapping["resources"]["cpu_threads"] = 2
    mapping["evaluation"]["holdout_fraction"] = 0
    mapping["learning"].update(initial_size=4, batch_size=2, cycles=1, budget=6)
    fake = write_fake_gnina(tmp_path / f"{name}_fake", behaviour)
    mapping["docking"] = {"receptor": str(receptor), "center": [1, 2, 3], "box_size": [10, 10, 10],
                          "exhaustiveness": 1, "n_poses": 1, "timeout": 60, **fake}
    return mapping, Path(fake["gnina_image"]).resolve().parent


def test_gnina_rescoring_records_extra_columns_manifest_and_failed_rows(tmp_path):
    probe, _ = fake_docking_mapping(tmp_path, "probe")
    failing_id = initial_sample_ids(probe)[0]
    mapping, fake = fake_docking_mapping(tmp_path, "rescored", {"affinity_from_name": True,
                                                                  "by_directory": {f"{failing_id}_": "exit"}})
    run_cli(mapping, tmp_path / "rescored.yaml")
    output = Path(mapping["output_dir"])
    observations = pd.read_csv(output / "observations.csv", float_precision="round_trip")
    assert list(observations.columns) == LEGACY_OBSERVATION_COLUMNS + RESCORING_COLUMNS
    for path in [output / "final_results.csv", *sorted(output.glob("cycle_*/queries.csv"))]:
        assert list(pd.read_csv(path).columns) == LEGACY_OBSERVATION_COLUMNS + RESCORING_COLUMNS, path
    assert len(list(output.glob("cycle_*/queries.csv"))) == 2
    summary = json.loads((output / "summary.json").read_text())
    assert (summary["attempted"], summary["successful"], summary["stop_reason"]) == (6, 5, "budget")
    failed = observations[observations.status == "failed"]
    assert failed.compound_id.tolist() == [failing_id]
    assert failed[["score", *RESCORING_COLUMNS]].isna().all(axis=None)
    assert "GNINA exited 3" in failed.error.item()
    with (output / "observations.csv").open(newline="") as handle:
        failed_row = next(row for row in csv.reader(handle) if row[0] == failing_id)
    assert failed_row[-3:] == ["", "", ""] and failed_row[5] == ""
    succeeded = observations[observations.status == "ok"]
    assert (succeeded.score == -succeeded.cnn_affinity).all()
    assert succeeded.score.nunique() > 1
    assert (succeeded.docking_energy == -7.25).all() and (succeeded.cnn_score == 0.8160941601).all()
    assert succeeded.pose.map(lambda p: Path(p).name == "rescore_pose.sdf").all()
    final = pd.read_csv(output / "final_results.csv", float_precision="round_trip")
    assert failing_id not in set(final.compound_id)
    assert final.score.tolist() == sorted(final.score.tolist())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["rescoring"] == {
        "engine": "gnina", "image": str(fake / "gnina.sif"), "image_sha256": sha256_file(fake / "gnina.sif"),
        "apptainer": str(fake / "apptainer"), "apptainer_version": "apptainer version 1.5.5-1",
        "version": "gnina v1.3.3 master:6fe1ce2   Built Jun 30 2026.",
        "cnn_model": ["dense_1_3", "dense_1_3_PT_KD_3", "crossdock_default2018_KD_4"], "input_format": "sdf"}
    resolved = yaml.safe_load((output / "config.resolved.yaml").read_text())["docking"]
    assert (resolved["rescoring"], resolved["gnina_image"]) == ("gnina", str(fake / "gnina.sif"))


def test_rescoring_off_keeps_columns_manifest_requests_and_pose_files(tmp_path):
    mapping, _ = fake_docking_mapping(tmp_path, "plain")
    for name in ("rescoring", "gnina_image", "apptainer_executable"):
        del mapping["docking"][name]
    run_cli(mapping, tmp_path / "plain.yaml")
    output = Path(mapping["output_dir"])
    for path in [output / "observations.csv", output / "final_results.csv",
                 *sorted(output.glob("cycle_*/queries.csv"))]:
        assert list(pd.read_csv(path).columns) == LEGACY_OBSERVATION_COLUMNS, path
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed" and "rescoring" not in manifest
    resolved = yaml.safe_load((output / "config.resolved.yaml").read_text())["docking"]
    assert (resolved["rescoring"], resolved["gnina_image"], resolved["apptainer_executable"]) == (
        "none", None, None)
    assert manifest["config"]["docking"] == resolved
    compounds = sorted(output.glob("cycle_*/poses/*"))
    assert len(compounds) == 6
    for directory in compounds:
        request = json.loads((directory / "request.json").read_text())
        assert not {"rescoring", "gnina_image", "apptainer_executable"} & set(request)
        assert not {"rescore_pose.pdbqt", "rescore_pose.sdf", "gnina_scored.sdf", "gnina.log"} & {
            p.name for p in directory.iterdir()}
        assert Path(json.loads((directory / "result.json").read_text())["pose"]).name == "best_poses.pdbqt"


def test_gnina_version_mismatch_stops_the_run_before_any_docking_budget(tmp_path):
    mapping, fake = fake_docking_mapping(tmp_path, "mismatch",
                                         {"version_output": "gnina v1.3.2 master:abcdef0   Built Jan 1 2026."})
    completed = run_cli(mapping, tmp_path / "mismatch.yaml", expected=1)
    assert "1.3.3" in completed.stderr
    output = Path(mapping["output_dir"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "1.3.3" in manifest["error"]
    assert not list(output.glob("cycle_*")) and not (output / "observations.csv").exists()
    assert "--score_only" not in (fake / "calls.jsonl").read_text()


@pytest.mark.integration
def test_real_gnina_rescoring_run_is_reproducible_and_worker_invariant(tmp_path):
    """Dock with Vina, rescore with the pinned GNINA image; repeat and parallel runs must match."""
    from active_learning_docking.experiments import compare_runs

    if os.environ.get("ALD_RUN_DOCKING_TESTS") != "1":
        pytest.skip("Set ALD_RUN_DOCKING_TESTS=1 to run native molecular docking")
    image, apptainer = os.environ.get("ALD_GNINA_IMAGE"), os.environ.get("ALD_APPTAINER")
    if not image or not apptainer:
        pytest.skip("Set ALD_GNINA_IMAGE (GNINA 1.3.3 SIF) and ALD_APPTAINER (Apptainer executable)")
    pytest.importorskip("vina")
    pytest.importorskip("meeko")
    mapping = yaml.safe_load((ROOT / "examples/docking.yaml").read_text())
    mapping["data"]["path"] = str(ROOT / "inputs/candidates.csv")
    mapping["docking"].update(receptor=str(ROOT / "examples/data/receptor.pdbqt"),
                              reference_ligand=str(ROOT / "inputs/reference.pdb"), rescoring="gnina",
                              gnina_image=image, apptainer_executable=apptainer)
    runs = {}
    for name, workers in (("first", 1), ("repeat", 1), ("parallel", 2)):
        mapping["output_dir"] = str(tmp_path / name)
        mapping["docking"]["workers"] = workers
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
        completed = subprocess.run([sys.executable, "-m", "active_learning_docking", "run", "--config", str(path)],
                                   cwd=ROOT, capture_output=True, text=True, timeout=3600)
        assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
        runs[name] = tmp_path / name
    first = runs["first"]
    observations = pd.read_csv(first / "observations.csv", float_precision="round_trip")
    assert list(observations.columns) == LEGACY_OBSERVATION_COLUMNS + RESCORING_COLUMNS
    succeeded = observations[observations.status == "ok"]
    assert len(succeeded) >= 3
    assert (succeeded.score == -succeeded.cnn_affinity).all()
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["rescoring"]["version"].startswith("gnina v1.3.3 ")
    assert len(manifest["rescoring"]["image_sha256"]) == 64
    for other in ("repeat", "parallel"):
        report = compare_runs(first, runs[other])
        assert report["passed"], (other, report["checks"])
        manifests = [json.loads((run / "manifest.json").read_text()) for run in (first, runs[other])]
        sdf = [{k: v for k, v in m["artifact_sha256"].items() if k.endswith(".sdf")} for m in manifests]
        assert sdf[0] and sdf[0] == sdf[1], other
        assert manifests[0]["rescoring"] == manifests[1]["rescoring"]
