"""Library store equivalence with in-memory loading, packed fingerprints and large-pool paths."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from active_learning_docking.data import LibraryValidationError, featurize, load_library
from active_learning_docking.model import TabMRegressor
from active_learning_docking.store import Library, PackedFingerprints, build_library_store, open_library_store
from tests.test_data import write_library
from tests.test_pipeline import run_cli, tiny_mapping

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "examples/replay.csv"


def test_packed_fingerprints_roundtrip_views_and_bounds():
    rng = np.random.default_rng(3)
    for n_bits in (13, 16, 256):
        dense = rng.integers(0, 2, size=(9, n_bits)).astype(np.uint8)
        packed = PackedFingerprints.from_dense(dense)
        assert packed.shape == (9, n_bits) and packed.packed.shape[1] == -(-n_bits // 8)
        np.testing.assert_array_equal(packed.dense(), dense)
        view = packed.take([7, 2, 2])
        np.testing.assert_array_equal(view[0:3], dense[[7, 2, 2]])
        np.testing.assert_array_equal(view.take([2, 0])[[0, 1]], dense[[2, 7]])
        assert view[1:2].dtype == np.uint8 and view[1:2].flags["C_CONTIGUOUS"]
    with pytest.raises(IndexError):
        packed.take([9])
    with pytest.raises(IndexError):
        packed[[-1]]
    mask = np.zeros(9, dtype=bool)
    mask[[4, 8]] = True
    np.testing.assert_array_equal(packed[mask], dense[[4, 8]])
    np.testing.assert_array_equal(packed.take(mask).dense(), dense[[4, 8]])
    with pytest.raises(IndexError, match="mask length"):
        packed[mask[:5]]
    with pytest.raises(ValueError, match="binary"):
        PackedFingerprints.from_dense(np.array([[0, 2]]))


def test_model_streams_lazy_views_identically_to_dense_input():
    rng = np.random.default_rng(11)
    dense = rng.integers(0, 2, size=(60, 40)).astype(np.uint8)
    y = dense[:, :5].sum(axis=1) - 2.0
    packed = PackedFingerprints.from_dense(dense)
    parameters = dict(n_blocks=1, d_block=16, k=4, epochs=4, batch_size=8, patience=2, seed=5, device="cpu")
    lazy = TabMRegressor(**parameters).fit(packed.take(np.arange(60)), y)
    plain = TabMRegressor(**parameters).fit(dense, y)
    assert lazy.history_ == plain.history_
    np.testing.assert_array_equal(lazy.predict_members(packed.take([3, 1, 59])), plain.predict_members(dense[[3, 1, 59]]))
    streamed = TabMRegressor(**parameters, inference_batch_size=7).fit(packed, y)
    mean, std = streamed.predict(packed.take(np.arange(60)))
    reference = plain.predict(dense)
    np.testing.assert_allclose(mean, reference[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(std, reference[1], rtol=0, atol=1e-6)


@pytest.mark.parametrize("workers,chunk_size", [(1, 20_000), (2, 7)])
def test_store_matches_in_memory_library_regardless_of_worker_layout(tmp_path, workers, chunk_size):
    metadata = build_library_store(REPLAY, tmp_path / "store", score_column="synthetic_score", n_bits=256,
                                   workers=workers, chunk_size=chunk_size)
    library, features, opened = open_library_store(tmp_path / "store")
    frame, rejected = load_library(REPLAY, score_column="synthetic_score")
    assert opened == metadata and metadata["n_compounds"] == len(frame) == len(library)
    assert metadata["n_input_records"] == 48 and metadata["n_rejected"] == len(rejected) == 0
    np.testing.assert_array_equal(library.ids(), frame.compound_id.to_numpy())
    np.testing.assert_array_equal(library.smiles(), frame.smiles.to_numpy())
    np.testing.assert_array_equal(library.scores(), frame.score.to_numpy())
    np.testing.assert_array_equal(features.dense(), featurize(frame.smiles.tolist(), n_bits=256))
    source = np.load(tmp_path / "store/source.npy")
    np.testing.assert_array_equal(source[:, 1], frame.source_row.to_numpy())
    assert list(library.smiles_column) == frame.smiles.tolist()
    assert library.frame([0, 2]).compound_id.tolist() == frame.compound_id.iloc[[0, 2]].tolist()
    for name, info in metadata["files"].items():
        assert (tmp_path / "store" / name).stat().st_size == info["bytes"]
    scores = tmp_path / "store/scores.npy"
    original = scores.read_bytes()
    scores.write_bytes(original[:-8] + bytes(8))  # Same size, different content.
    with pytest.raises(ValueError, match="checksum mismatch"):
        open_library_store(tmp_path / "store")
    open_library_store(tmp_path / "store", verify=False)
    scores.write_bytes(original)


def test_store_deduplication_and_rejections_match_load_library(tmp_path):
    rows = [("z", "CCO", -5), ("b", "OCC", -5), ("b", "OCC", -5), ("a", "OCC", -5), ("q", "not-a-smiles", 1),
            ("k", "CCN", -7), ("", "CC", -1), ("m", "CCN", -7)]
    data = write_library(tmp_path, rows)
    frame, rejected = load_library(data, invalid_policy="skip")
    metadata = build_library_store(data, tmp_path / "store", score_column="docking_score", invalid_policy="skip",
                                   n_bits=64)
    library, _, _ = open_library_store(tmp_path / "store")
    assert library.ids().tolist() == frame.compound_id.tolist() == ["a", "k"]
    stored = pd.read_csv(tmp_path / "store/rejected_rows.csv", keep_default_na=False, dtype=str)
    expected = rejected.astype(str)
    assert stored[["compound_id", "source_row", "reason", "kept_compound_id"]].values.tolist() == \
        expected[["compound_id", "source_row", "reason", "kept_compound_id"]].values.tolist()
    assert metadata["n_rejected"] == len(rejected) == 6 and metadata["n_input_records"] == 8
    assert stored.reason.tolist().count("duplicate_structure") == 3
    assert set(stored.loc[stored.reason.str.startswith("duplicate_"), "kept_compound_id"]) == {"a", "k"}
    kept_all = build_library_store(data, tmp_path / "nodedup", score_column="docking_score", invalid_policy="skip",
                                   deduplicate=False, n_bits=64)
    assert kept_all["n_compounds"] == 5


@pytest.mark.parametrize("rows,reason", [
    ([("z", "CCO", -5), ("a", "OCC", -5), ("z", "CCN", -5)], "conflicting_compound_id"),
    ([("z", "CCO", -5), ("a", "OCC", -5), ("b", "OCC", -6)], "conflicting_structure_score"),
    ([("z", "bad", -5)], "invalid_smiles"),
])
def test_store_builder_keeps_fatal_validation(tmp_path, rows, reason):
    data = write_library(tmp_path, rows)
    with pytest.raises(LibraryValidationError, match=reason) as error:
        build_library_store(data, tmp_path / "store", score_column="docking_score")
    assert reason in error.value.rejections.reason.tolist()
    assert not (tmp_path / "store").exists()


def test_store_builder_reports_an_empty_library_and_cleans_up(tmp_path):
    data = write_library(tmp_path, [("a", "bad", -5), ("b", "worse", -6)])
    (tmp_path / "store").mkdir()
    with pytest.raises(LibraryValidationError, match="No valid compounds") as error:
        build_library_store(data, tmp_path / "store", score_column="docking_score", invalid_policy="skip")
    assert error.value.rejections.reason.tolist() == ["invalid_smiles", "invalid_smiles"]
    assert (tmp_path / "store").is_dir() and not any((tmp_path / "store").iterdir())


def test_store_refuses_existing_outputs_and_mismatched_configuration(tmp_path):
    (tmp_path / "store").mkdir()
    (tmp_path / "store/library.json").write_text("{}")
    with pytest.raises(FileExistsError):
        build_library_store(REPLAY, tmp_path / "store", score_column="synthetic_score")
    build_library_store(REPLAY, tmp_path / "fresh", score_column="synthetic_score", n_bits=256)
    output = tmp_path / "run"
    mapping = tiny_mapping(output, tmp_path / "fresh")
    mapping["data"]["fingerprint_bits"] = 128
    completed = run_cli(mapping, tmp_path / "mismatch.yaml", expected=1)
    assert "differ from configuration" in completed.stderr
    mapping = tiny_mapping(tmp_path / "run2", tmp_path / "fresh")
    mapping["data"]["fingerprint_bits"] = 256
    mapping["data"]["deduplicate"] = False
    completed = run_cli(mapping, tmp_path / "mismatch2.yaml", expected=1)
    assert "deduplicate" in completed.stderr
    mapping = tiny_mapping(tmp_path / "run3", tmp_path / "fresh")
    mapping["data"].update(fingerprint_bits=256, score_column="docking_score")
    completed = run_cli(mapping, tmp_path / "mismatch3.yaml", expected=1)
    assert "score_column" in completed.stderr


def test_store_run_reproduces_csv_run_and_omits_library_copy(tmp_path):
    from active_learning_docking.experiments import compare_runs

    csv_output = tmp_path / "csv_run"
    run_cli(tiny_mapping(csv_output), tmp_path / "csv.yaml")
    build_library_store(REPLAY, tmp_path / "store", score_column="synthetic_score", n_bits=128)
    store_output = tmp_path / "store_run"
    run_cli(tiny_mapping(store_output, tmp_path / "store"), tmp_path / "store.yaml")
    report = compare_runs(csv_output, store_output)
    assert report["passed"], report["checks"]
    assert (csv_output / "library.csv").exists() and not (store_output / "library.csv").exists()
    manifest = json.loads((store_output / "manifest.json").read_text())
    assert manifest["library_store"]["n_compounds"] == 48
    assert list(manifest["input_sha256"]) == [str(tmp_path / "store/library.json")]


def test_prediction_tables_are_optional_and_split_requires_a_holdout(tmp_path):
    output = tmp_path / "lean"
    mapping = tiny_mapping(output)
    mapping["evaluation"].update(holdout_fraction=0, save_predictions=False)
    run_cli(mapping, tmp_path / "lean.yaml")
    assert not (output / "predictions.csv").exists() and not (output / "split.csv").exists()
    assert not list(output.glob("cycle_*/pool_predictions_before_acquisition.csv"))
    observations = pd.read_csv(output / "observations.csv")
    assert observations.prediction_before_query.notna().sum() == len(observations) - 6
    full = tmp_path / "full"
    mapping = tiny_mapping(full)
    mapping["evaluation"]["holdout_fraction"] = 0
    run_cli(mapping, tmp_path / "full.yaml")
    assert (full / "observations.csv").read_bytes() == (output / "observations.csv").read_bytes()
    assert (full / "predictions.csv").exists() and not (full / "split.csv").exists()


def test_cli_prepare_library_builds_a_store_under_the_resource_policy(tmp_path):
    config = tmp_path / "prepare.yaml"
    config.write_text(yaml.safe_dump({
        "seed": 3, "output_dir": str(tmp_path / "unused"),
        "resources": {"cpu_threads": 2, "device": "cpu", "gpu_ids": [0, 1]},
        "data": {"path": str(REPLAY), "score_column": "synthetic_score", "fingerprint_bits": 128},
    }))
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "1"}
    completed = subprocess.run(
        [sys.executable, "-m", "active_learning_docking", "prepare-library", "--config", str(config),
         "--output", str(tmp_path / "store"), "--chunk-size", "5"],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    printed = json.loads(completed.stdout)
    assert printed["n_compounds"] == 48 and printed["fingerprint"]["n_bits"] == 128
    library, features, metadata = open_library_store(tmp_path / "store")
    assert metadata["settings"]["workers"] == 2 and features.shape == (48, 128)
    assert isinstance(library, Library) and library.origin == "store"
