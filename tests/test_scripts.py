"""Generation scripts reject output collisions before changing existing evidence."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from active_learning_docking import data, docking


ROOT = Path(__file__).resolve().parents[1]


def script_module(name, path=None):
    spec = importlib.util.spec_from_file_location(name, path or ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(directory):
    """Capture both evidence bytes and link/directory topology."""
    return {
        str(path.relative_to(directory)): (
            ("link", os.readlink(path)) if path.is_symlink() else
            ("file", hashlib.sha256(path.read_bytes()).hexdigest()) if path.is_file() else ("directory",)
        )
        for path in directory.rglob("*")
    }


def assert_preflight_failure(module, arguments, directory, monkeypatch):
    before = snapshot(directory)
    monkeypatch.setattr(sys, "argv", [module.__file__, *map(str, arguments)])

    def unexpected(*args, **kwargs):
        pytest.fail("generation or input evaluation started before collision validation")

    monkeypatch.setattr(module, "version", unexpected)
    monkeypatch.setattr(data, "load_library", unexpected)
    monkeypatch.setattr(docking, "DockingOracle", unexpected)
    if hasattr(module, "subprocess"):
        monkeypatch.setattr(module.subprocess, "run", unexpected)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert snapshot(directory) == before


@pytest.mark.parametrize("suffix", [".pdbqt", ".prepared.pdb", ".preparation.log", ".provenance.json"])
@pytest.mark.parametrize("kind", ["file", "link", "dangling_link"])
def test_receptor_preflights_every_generated_file(tmp_path, monkeypatch, suffix, kind):
    source = tmp_path / "input.pdb"
    source.write_text("original input")
    output = tmp_path / "protein.pdbqt"
    target = output.with_suffix(suffix)
    if kind == "file":
        target.write_text("original preparation evidence")
    else:
        target.symlink_to(source if kind == "link" else tmp_path / "missing")
    assert_preflight_failure(script_module("prepare_receptor"), [
        "--input", source, "--output", output, "--repair-missing-atoms",
    ], tmp_path, monkeypatch)


@pytest.mark.parametrize("suffix", [".pdbqt", ".prepared.pdb", ".preparation.log", ".provenance.json"])
def test_receptor_generated_paths_cannot_alias_input(tmp_path, monkeypatch, suffix):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    source = actual / f"protein{suffix}"
    source.write_text("original input via directory alias")
    assert_preflight_failure(script_module("prepare_receptor"), [
        "--input", source, "--output", alias / "protein.pdbqt", "--repair-missing-atoms",
    ], tmp_path, monkeypatch)


@pytest.mark.parametrize("target_name", [
    "scores.csv", "scores.provenance.json", "artifacts/input_rejections.csv", "artifacts/records.json",
    "artifacts/failures.json", "artifacts/poses", "artifacts",
])
@pytest.mark.parametrize("kind", ["file", "dangling_link"])
def test_replay_preflights_all_outputs(tmp_path, monkeypatch, target_name, kind):
    source = tmp_path / "input.csv"
    source.write_text("Catalog ID,smiles\na,CCO\n")
    target = tmp_path / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "file":
        target.write_text("original replay evidence")
    else:
        target.symlink_to(tmp_path / "missing")
    assert_preflight_failure(script_module("build_replay"), [
        "--input", source, "--output", tmp_path / "scores.csv", "--artifacts", tmp_path / "artifacts", "--count", 1,
    ], tmp_path, monkeypatch)


@pytest.mark.parametrize("output_name,artifact_name", [
    ("artifacts/input_rejections.csv", "artifacts"), ("artifacts/records.json", "artifacts"),
    ("artifacts/failures.json", "artifacts"), ("artifacts/poses", "artifacts"),
    ("artifacts/poses/scores.csv", "artifacts"), ("artifacts", "artifacts"),
    ("scores.csv", "scores.csv/artifacts"),
])
def test_replay_rejects_collisions_between_new_generated_paths(tmp_path, monkeypatch, output_name, artifact_name):
    source = tmp_path / "input.csv"
    source.write_text("Catalog ID,smiles\na,CCO\n")
    assert_preflight_failure(script_module("build_replay"), [
        "--input", source, "--output", tmp_path / output_name, "--artifacts", tmp_path / artifact_name, "--count", 1,
    ], tmp_path, monkeypatch)


@pytest.mark.parametrize("input_argument", ["--input", "--receptor", "--reference"])
def test_replay_provenance_cannot_alias_any_input(tmp_path, monkeypatch, input_argument):
    source = tmp_path / "input.csv"
    source.write_text("Catalog ID,smiles\na,CCO\n")
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    collision = actual / "scores.provenance.json"
    collision.write_text("input evidence")
    assert_preflight_failure(script_module("build_replay"), [
        "--input", source, input_argument, collision, "--output", alias / "scores.csv",
        "--artifacts", tmp_path / "artifacts", "--count", 1,
    ], tmp_path, monkeypatch)


def test_replay_rejects_directory_input_before_any_evaluation(tmp_path, monkeypatch):
    source = tmp_path / "inputs"
    source.mkdir()
    (source / "input.csv").write_text("Catalog ID,smiles\na,CCO\n")
    assert_preflight_failure(script_module("build_replay"), [
        "--input", source, "--output", tmp_path / "scores.csv", "--artifacts", tmp_path / "artifacts", "--count", 1,
    ], tmp_path, monkeypatch)


def test_replay_output_inside_new_artifacts_and_imported_oracle_hash(tmp_path, monkeypatch):
    # Relocating the generator leaves no adjacent docking.py to hash.
    generator = tmp_path / "relocated" / "build_replay.py"
    generator.parent.mkdir()
    shutil.copyfile(ROOT / "scripts/build_replay.py", generator)
    module = script_module("relocated_replay", generator)
    source = tmp_path / "input.csv"
    source.write_text("Catalog ID,smiles\na,CCO\n")
    receptor, reference = tmp_path / "receptor.pdbqt", tmp_path / "reference.pdb"
    receptor.write_text("receptor evidence")
    reference.write_text("reference evidence")
    artifacts = tmp_path / "artifacts"
    output = artifacts / "nested" / "scores.csv"

    class Oracle:
        def __init__(self, configuration, seed, cpu):
            self.parameters, self.timeout = configuration.copy(), configuration["timeout"]

        def evaluate(self, cid, smiles, output_dir):
            output_dir.mkdir(parents=True)
            pose = output_dir / "pose.pdbqt"
            pose.write_text("deterministic pose")
            return {"cid": cid, "status": "ok", "score": -4.5, "pose": str(pose)}

    monkeypatch.setattr(docking, "DockingOracle", Oracle)
    monkeypatch.setattr(module, "version", lambda name: "test-version")
    if hasattr(os, "sched_setaffinity"):
        monkeypatch.setattr(os, "sched_setaffinity", lambda *args: None)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    arguments = ["--input", source, "--receptor", receptor, "--reference", reference,
                 "--output", output, "--artifacts", artifacts, "--count", 1]
    monkeypatch.setattr(sys, "argv", [str(generator), *map(str, arguments)])
    module.main()
    assert output.read_text() == "Catalog ID,smiles,docking_score\na,CCO,-4.5\n"
    provenance = json.loads(output.with_suffix(".provenance.json").read_text())
    assert provenance["oracle_sha256"] == hashlib.sha256(Path(docking.__file__).read_bytes()).hexdigest()
    assert provenance["generator_sha256"] == hashlib.sha256(generator.read_bytes()).hexdigest()
    assert provenance["succeeded"] == 1 and provenance["failed"] == 0
    assert json.loads((artifacts / "failures.json").read_text()) == []
    assert_preflight_failure(module, arguments, tmp_path, monkeypatch)
