"""Fresh-process reproducibility, paired controls and archive-integrity tests."""

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from active_learning_docking.config import from_dict
from active_learning_docking.experiments import benchmark, compare_runs, reproduce
from active_learning_docking.provenance import artifact_hashes, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def small_config(output):
    return from_dict({
        "seed": 42, "output_dir": str(output),
        "resources": {"cpu_threads": 1, "device": "cpu"},
        "data": {"path": str(ROOT / "examples/replay.csv"), "score_column": "synthetic_score", "fingerprint_bits": 128},
        "model": {"models": ["tabm"], "n_blocks": 1, "d_block": 16, "k": 4, "epochs": 1, "batch_size": 8},
        "learning": {"initial_size": 6, "batch_size": 3, "cycles": 1, "budget": 9},
        "evaluation": {"holdout_fraction": 0.2, "top_k": 3},
    })


@pytest.fixture(scope="module")
def repeated_runs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("reproduce")
    report = reproduce(small_config(directory / "unused"), directory / "repeats")
    return directory / "repeats", report


def test_independent_processes_reproduce_exact_queries_metrics_and_predictions(repeated_runs):
    root, report = repeated_runs
    assert report["passed"]
    assert report["comparison"] == "exact"
    for required in ("observations.csv", "predictions.csv", "metrics.csv", "training_history.csv", "summary.json"):
        assert report["checks"][required]
    assert json.loads((root / "reproducibility.json").read_text())["passed"]
    for name in ("repeat_a", "repeat_b"):
        assert (root / f"{name}.log").exists()
        assert (root / name / "model/tabm_weights.pt").exists()


def test_changed_artifact_is_rejected_even_when_scientific_outputs_match(repeated_runs, tmp_path):
    root, _ = repeated_runs
    altered = tmp_path / "altered"
    shutil.copytree(root / "repeat_b", altered)
    with (altered / "model/tabm_weights.pt").open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="integrity mismatch"):
        compare_runs(root / "repeat_a", altered)


def test_unlisted_artifact_is_rejected(repeated_runs, tmp_path):
    root, _ = repeated_runs
    altered = tmp_path / "altered"
    shutil.copytree(root / "repeat_b", altered)
    (altered / "unexpected.csv").write_text("compound_id,score\na,-1000\n")
    with pytest.raises(ValueError, match="integrity|inventory|artifact"):
        compare_runs(root / "repeat_a", altered)


def test_prediction_difference_in_valid_archives_fails_exact_comparison(repeated_runs, tmp_path):
    root, _ = repeated_runs
    altered = tmp_path / "altered"
    shutil.copytree(root / "repeat_b", altered)
    path = altered / "predictions.csv"
    predictions = pd.read_csv(path, float_precision="round_trip")
    predictions.loc[0, "prediction"] += 1e-8
    predictions.to_csv(path, index=False)
    manifest_path = altered / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"]["predictions.csv"] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))
    report = compare_runs(root / "repeat_a", altered)
    assert not report["passed"]
    assert not report["checks"]["predictions.csv"]


@pytest.mark.parametrize("name,contents", [
    ("cycle_999/queries.csv", "compound_id,score\na,-5\n"),
    ("cycle_999/metrics.json", '{"cycle": 999}\n'),
    ("cycle_000/metrics.json", '{"altered_metric": 1}\n'),
    ("cycle_000/holdout_predictions.csv", "renamed_header\n"),
    ("training_history.csv", "reordered,headers\n"),
])
def test_cycle_inventory_and_metrics_are_compared_symmetrically(repeated_runs, tmp_path, name, contents):
    root, _ = repeated_runs
    altered = tmp_path / "altered"
    shutil.copytree(root / "repeat_b", altered)
    path = altered / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(contents)
    manifest_path = altered / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"] = artifact_hashes(altered)
    manifest_path.write_text(json.dumps(manifest))
    for first, second in ((root / "repeat_a", altered), (altered, root / "repeat_a")):
        report = compare_runs(first, second)
        assert not report["passed"]
        assert not report["checks"][name]


@pytest.mark.parametrize("left,right", [
    ("old_header\n", "new_header\n"),
    ("id,score\na,-5\n", "score,id\n-5,a\n"),
    ("id,score,score\na,-4,-5\n", "id,score,score\na,-6,-5\n"),
])
def test_csv_headers_and_all_fields_are_part_of_comparison(repeated_runs, tmp_path, left, right):
    root, _ = repeated_runs
    copies = []
    for index, contents in enumerate((left, right)):
        directory = tmp_path / str(index)
        shutil.copytree(root / "repeat_a", directory)
        (directory / "cycle_000/header_check.csv").write_text(contents)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_sha256"] = artifact_hashes(directory)
        manifest_path.write_text(json.dumps(manifest))
        copies.append(directory)
    report = compare_runs(*copies)
    assert not report["passed"]
    assert not report["checks"]["cycle_000/header_check.csv"]


@pytest.mark.parametrize("left,right,passed", [
    ("$$$$\n", "$$$$\n", True),
    ("> <minimizedAffinity>\n-7.1\n$$$$\n", "> <minimizedAffinity>\n-7.2\n$$$$\n", False),
])
def test_sdf_pose_bytes_are_compared(repeated_runs, tmp_path, left, right, passed):
    root, _ = repeated_runs
    copies = []
    for index, contents in enumerate((left, right)):
        directory = tmp_path / str(index)
        shutil.copytree(root / "repeat_a", directory)
        pose = directory / "cycle_000/poses/z1_0123456789abcdef"
        pose.mkdir(parents=True)
        (pose / "gnina_scored.sdf").write_text(contents)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_sha256"] = artifact_hashes(directory)
        manifest_path.write_text(json.dumps(manifest))
        copies.append(directory)
    for first, second in (copies, copies[::-1]):
        report = compare_runs(first, second)
        assert report["passed"] is passed
        assert report["checks"]["all_pose_bytes"] is passed


def test_sdf_pose_present_in_only_one_run_fails(repeated_runs, tmp_path):
    root, _ = repeated_runs
    altered = tmp_path / "altered"
    shutil.copytree(root / "repeat_a", altered)
    pose = altered / "cycle_000/poses/z1_0123456789abcdef"
    pose.mkdir(parents=True)
    (pose / "rescore_pose.sdf").write_text("$$$$\n")
    manifest_path = altered / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"] = artifact_hashes(altered)
    manifest_path.write_text(json.dumps(manifest))
    report = compare_runs(root / "repeat_a", altered)
    assert not report["passed"]
    assert report["checks"] == {**compare_runs(root / "repeat_a", root / "repeat_b")["checks"],
                                "all_pose_bytes": False}


def test_incomplete_archive_is_not_reproducible(repeated_runs, tmp_path):
    root, _ = repeated_runs
    altered = tmp_path / "altered"
    shutil.copytree(root / "repeat_b", altered)
    path = altered / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = "failed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not completed"):
        compare_runs(root / "repeat_a", altered)


def test_paired_seed_benchmark_uses_matched_initial_sets_and_budgets(tmp_path):
    output = tmp_path / "benchmark"
    report = benchmark(small_config(tmp_path / "unused"), output, seeds=[17, 42], strategies=["random", "lcb"])
    results = pd.read_csv(output / "paired_results.csv")
    assert len(results) == 4
    assert results.attempted.tolist() == [9, 9, 9, 9]
    assert report["aggregate"]["lcb"]["discovery_top_k_recall"]["n_seeds"] == 2
    first_selections = []
    for seed in (17, 42):
        random_output, lcb_output = [output / f"{strategy}_seed_{seed}" for strategy in ("random", "lcb")]
        assert (random_output / "split.csv").read_bytes() == (lcb_output / "split.csv").read_bytes()
        random_initial = pd.read_csv(random_output / "cycle_000/queries.csv").compound_id.tolist()
        lcb_initial = pd.read_csv(lcb_output / "cycle_000/queries.csv").compound_id.tolist()
        assert random_initial == lcb_initial
        first_selections.append(random_initial)
    assert first_selections[0] != first_selections[1]


@pytest.mark.parametrize("seeds,strategies", [([42], ["lcb"]), ([42, 42], ["lcb"]), ([1, 2], []), ([1, 2], ["lcb", "lcb"])])
def test_invalid_benchmark_design_fails_before_launch(tmp_path, seeds, strategies):
    with pytest.raises(ValueError):
        benchmark(small_config(tmp_path / "unused"), tmp_path / "bench", seeds, strategies)
    assert not (tmp_path / "bench").exists()


def test_reproduction_never_overwrites_previous_results(repeated_runs):
    root, _ = repeated_runs
    checksum = sha256_file(root / "reproducibility.json")
    with pytest.raises(FileExistsError):
        reproduce(small_config(root / "unused"), root)
    assert sha256_file(root / "reproducibility.json") == checksum
