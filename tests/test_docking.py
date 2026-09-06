"""Oracle failure semantics, resource bounds, and opt-in real docking integration."""

import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from active_learning_docking.docking import DockingOracle, _run_bounded, compound_seed, prepare_ligands, reference_center
from active_learning_docking import docking


ROOT = Path(__file__).resolve().parents[1]


def test_seeds_are_stable_order_independent_and_nonzero():
    expected = compound_seed(42, "a", "CCO")
    compound_seed(42, "b", "CC")
    assert compound_seed(42, "a", "CCO") == expected
    assert 0 < expected < 2**31
    assert len({expected, compound_seed(43, "a", "CCO"), compound_seed(42, "b", "CCO")}) == 3


def test_reference_center_excludes_hydrogens(tmp_path):
    path = tmp_path / "reference.pdb"
    lines = []
    for index, (element, x) in enumerate([("C", 1.0), ("O", 3.0), ("H", 100.0)], start=1):
        lines.append(f"HETATM{index:5d}  {element:2s}  UNL A   1    {x:8.3f}{2.:8.3f}{3.:8.3f}  1.00  0.00          {element:>2s}  ")
    path.write_text("\n".join(lines))
    assert reference_center(path) == [2.0, 2.0, 3.0]
    path.write_text("END\n")
    with pytest.raises(ValueError, match="no heavy"):
        reference_center(path)


def test_reference_center_reads_sdf_mol_and_mol2_heavy_atoms(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.AddHs(Chem.MolFromSmiles("CO"))
    AllChem.EmbedMolecule(molecule, randomSeed=1)
    conformer = molecule.GetConformer()
    heavy = [list(conformer.GetAtomPosition(a.GetIdx())) for a in molecule.GetAtoms() if a.GetAtomicNum() > 1]
    expected = [sum(p[i] for p in heavy) / len(heavy) for i in range(3)]
    for suffix in (".sdf", ".mol"):
        path = tmp_path / f"reference{suffix}"
        Chem.MolToMolFile(molecule, str(path))
        assert reference_center(path) == pytest.approx(expected, abs=1e-3)
    mol2 = tmp_path / "reference.mol2"
    mol2.write_text(
        "@<TRIPOS>MOLECULE\nref\n 3 2 0 0 0\nSMALL\nGASTEIGER\n\n@<TRIPOS>ATOM\n"
        "      1 C1          1.0000    0.0000    0.0000 C.3     1  UNL         0.0000\n"
        "      2 O1          3.0000    0.0000    0.0000 O.3     1  UNL         0.0000\n"
        "      3 H1          9.0000    0.0000    0.0000 H       1  UNL         0.0000\n"
        "@<TRIPOS>BOND\n     1     1     2    1\n     2     1     3    1\n"
    )
    assert reference_center(mol2) == pytest.approx([2.0, 0.0, 0.0])
    empty = tmp_path / "empty.sdf"
    empty.write_text("")
    with pytest.raises(ValueError, match="could not be parsed"):
        reference_center(empty)
    with pytest.raises(ValueError, match="reference_ligand must be one of"):
        reference_center(tmp_path / "reference.xyz")


@pytest.mark.parametrize("cpu", [0, -1, 10, True, 1.5])
def test_rejects_unbounded_or_invalid_cpu(cpu):
    with pytest.raises(ValueError, match="cpu"):
        DockingOracle({}, cpu=cpu)


def test_worker_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        _run_bounded([sys.executable, "-c", "import time; time.sleep(10)"], 0.1, os.environ.copy())


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-state inspection")
@pytest.mark.parametrize("worker_exits", [False, True])
def test_timeout_terminates_engine_descendants(tmp_path, worker_exits):
    marker = tmp_path / "engine.pid"
    script = (
        "import subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"Path({str(marker)!r}).write_text(str(child.pid)); "
        + ("sys.exit(0)" if worker_exits else "time.sleep(30)")
    )
    start = time.monotonic()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            _run_bounded([sys.executable, "-c", script], 1.0, os.environ.copy())
        assert time.monotonic() - start < 3
        state_path = Path(f"/proc/{marker.read_text()}/stat")
        for _ in range(50):
            try:
                if state_path.read_text().split()[2] == "Z":
                    return
            except (FileNotFoundError, ProcessLookupError):
                return  # The descendant disappeared between the existence check and the read.
            time.sleep(0.01)
        pytest.fail("engine descendant continued running after worker timeout")
    finally:
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
@pytest.mark.parametrize("original", [subprocess.TimeoutExpired("worker", 0.1), KeyboardInterrupt()])
@pytest.mark.parametrize("kill_error", [None, ProcessLookupError(), PermissionError()])
def test_cleanup_is_bounded_and_preserves_first_exception(monkeypatch, original, kill_error):
    from unittest.mock import Mock

    process = Mock(pid=12345)
    process.poll.return_value = 0
    process.communicate.side_effect = [original, subprocess.TimeoutExpired("cleanup", 1)]
    process.stdout.close.side_effect = OSError("pipe close failed")
    monkeypatch.setattr(docking.subprocess, "Popen", Mock(return_value=process))
    killpg = Mock(side_effect=kill_error)
    monkeypatch.setattr(docking.os, "killpg", killpg)
    with pytest.raises(type(original)) as error:
        _run_bounded(["worker"], 0.1, {})
    assert error.value is original
    killpg.assert_called_once_with(process.pid, signal.SIGKILL)
    assert [call.kwargs["timeout"] for call in process.communicate.call_args_list] == [0.1, 1.0]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_cleanup_timeout_does_not_wait_for_escaped_child(tmp_path):
    marker = tmp_path / "escaped.pid"
    script = (
        "import subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], start_new_session=True); "
        f"Path({str(marker)!r}).write_text(str(child.pid)); time.sleep(30)"
    )
    start = time.monotonic()
    try:
        with pytest.raises(subprocess.TimeoutExpired) as error:
            _run_bounded([sys.executable, "-c", script], 0.5, os.environ.copy())
        assert error.value.timeout == 0.5
        assert time.monotonic() - start < 3
    finally:
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("state", ["ok", "failed", "timeout", "interrupted"])
def test_compound_evidence_is_preserved_before_worker_reuse(tmp_path, monkeypatch, state):
    oracle = DockingOracle.__new__(DockingOracle)
    oracle.seed, oracle.timeout, oracle.parameters = 42, 0.1, {}

    def run(command, timeout, environment):
        directory = Path(command[-1]).parent
        (directory / "partial.pdbqt").write_text("original pose bytes")
        if state == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        if state == "interrupted":
            raise KeyboardInterrupt()
        result = {"status": state, "score": -4 if state == "ok" else None,
                  "pose": str(directory / "partial.pdbqt") if state == "ok" else None}
        Path(command[-1]).write_text(json.dumps(result))
        return subprocess.CompletedProcess(command, 0, "original worker log")

    monkeypatch.setattr(docking, "_run_bounded", run)
    if state == "interrupted":
        with pytest.raises(KeyboardInterrupt):
            oracle.evaluate("a", "CCO", tmp_path)
    else:
        record = oracle.evaluate("a", "CCO", tmp_path)
        assert record["status"] == ("failed" if state == "timeout" else state)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(docking, "_run_bounded", lambda *args: pytest.fail("worker must not launch"))
    with pytest.raises(FileExistsError):
        oracle.evaluate("a", "CCO", tmp_path)
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after


@pytest.mark.parametrize("backend,target", [
    (backend, target) for backend in ("vina", "qvina")
    for target in ("conformer_001.pdbqt", "poses_001.pdbqt", "best_poses.pdbqt")
] + [("qvina", "engine_001.log")])
@pytest.mark.parametrize("is_link", [False, True])
def test_internal_dock_checks_all_outputs_before_preparation(tmp_path, monkeypatch, backend, target, is_link):
    path = tmp_path / target
    if is_link:
        path.symlink_to(tmp_path / "absent")
    else:
        path.write_bytes(b"original evidence")
    monkeypatch.setattr(docking, "prepare_ligands", lambda *args: pytest.fail("preflight must run first"))
    with pytest.raises(FileExistsError):
        docking._dock({"directory": str(tmp_path), "backend": backend, "conformers": 2})
    assert list(tmp_path.iterdir()) == [path]
    assert path.is_symlink() if is_link else path.read_bytes() == b"original evidence"


def test_worker_uses_imported_module_despite_shadow_package_in_cwd(tmp_path, monkeypatch):
    shadow = tmp_path / "active_learning_docking"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("")
    (shadow / "docking.py").write_text("raise RuntimeError('wrong worker module')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    oracle = DockingOracle.__new__(DockingOracle)
    oracle.seed, oracle.timeout, oracle.parameters = 42, 10, {}
    result = oracle.evaluate("bad", "not-smiles", tmp_path / "output")
    assert result["status"] == "failed"
    assert "KeyError: 'conformers'" == result["error"], result
    assert "wrong worker module" not in result["error"]


def test_worker_refuses_existing_result_before_docking(tmp_path, monkeypatch):
    result = tmp_path / "result.json"
    result.write_text("original result")
    monkeypatch.setattr(docking, "_dock", lambda *args: pytest.fail("worker must preflight its result"))
    with pytest.raises(FileExistsError):
        docking._worker(str(tmp_path / "request.json"), str(result))
    assert result.read_text() == "original result"


def test_ligand_preparation_is_seeded_and_preserves_chemical_input():
    pytest.importorskip("meeko")
    first = prepare_ligands("CCO", 1729, 2)
    assert first == prepare_ligands("CCO", 1729, 2)
    assert len(first) == 2
    assert all("REMARK SMILES CCO" in ligand for ligand in first)
    with pytest.raises(ValueError, match="invalid"):
        prepare_ligands("not-smiles", 1729, 1)
    with pytest.raises(ValueError, match="disconnected"):
        prepare_ligands("CC.[Na+]", 1729, 1)


def _fake_oracle(tmp_path, energy="-4.125"):
    pytest.importorskip("meeko")
    receptor = tmp_path / "protein.pdbqt"
    receptor.write_text("ATOM      1  C   ALA A   1       1.000   2.000   3.000  1.00  0.00     0.000 C\n")
    executable = tmp_path / "fake qvina"
    executable.write_text(f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
                          "args=sys.argv[1:]\np=Path(args[args.index('--out')+1])\n"
                          "p.with_suffix('.args.json').write_text(json.dumps(args))\n"
                          f"p.write_text('REMARK VINA RESULT: {energy} 0 0\\nATOM      1  C\\n')\n")
    executable.chmod(0o755)
    return DockingOracle({"backend": "qvina", "executable": str(executable), "receptor": str(receptor),
                          "center": [1, 2, 3], "box_size": [10, 10, 10], "exhaustiveness": 1,
                          "n_poses": 1, "timeout": 30}, seed=42, cpu=1)


def test_qvina_arguments_pose_provenance_and_safe_compound_path(tmp_path):
    oracle = _fake_oracle(tmp_path)
    record = oracle.evaluate("../../escape; echo bad", "CCO", tmp_path / "output")
    assert record["status"] == "ok", record
    assert record["score"] == -4.125
    path = Path(record["pose"])
    assert path.is_relative_to(tmp_path / "output")
    args = json.loads(next(path.parent.glob("*.args.json")).read_text())
    assert args[args.index("--cpu") + 1] == "1"
    assert int(args[args.index("--seed") + 1]) > 0
    assert args[args.index("--center_x") + 1] == "1.0"
    assert (path.parent / "request.json").is_file()
    assert json.loads((path.parent / "result.json").read_text())["score"] == record["score"]


@pytest.mark.parametrize("energy", ["nan", "inf", "-inf"])
def test_nonfinite_engine_scores_are_failures(tmp_path, energy):
    record = _fake_oracle(tmp_path, energy).evaluate("test", "CCO", tmp_path / "output")
    assert record["status"] == "failed"
    assert record["score"] is None
    assert "non-finite" in record["error"]


def test_invalid_ligand_is_a_structured_failure(tmp_path):
    record = _fake_oracle(tmp_path).evaluate("bad", "not-smiles", tmp_path / "output")
    assert record["status"] == "failed"
    assert record["score"] is None and record["pose"] is None
    assert "invalid" in record["error"]


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["vina", "qvina"])
def test_real_docking_reproducibility(tmp_path, backend):
    """Dock an actual supplied candidate twice with the same seed and receptor."""
    if os.environ.get("ALD_RUN_DOCKING_TESTS") != "1":
        pytest.skip("Set ALD_RUN_DOCKING_TESTS=1 to run native molecular docking")
    pytest.importorskip("vina")
    pytest.importorskip("meeko")
    receptor = ROOT / "examples/data/receptor.pdbqt"
    assert receptor.is_file(), "Run scripts/prepare_receptor.py as documented"
    configuration = {"backend": backend, "receptor": str(receptor), "reference_ligand": str(ROOT / "inputs/reference.pdb"),
                     "box_size": [20, 20, 20], "exhaustiveness": 1, "n_poses": 1, "conformers": 1,
                     "timeout": 120, "executable": str(ROOT / "bin/qvina2.1")}
    smiles = "O=C(c1n[nH]c(=O)c2ccccc12)N1CCC(c2ccccc2)C1"
    first = DockingOracle(configuration, seed=42, cpu=1).evaluate("Z942299110", smiles, tmp_path / "first")
    second = DockingOracle(configuration, seed=42, cpu=1).evaluate("Z942299110", smiles, tmp_path / "second")
    assert first["status"] == second["status"] == "ok", (first, second)
    assert math.isfinite(first["score"]) and first["score"] < 0
    assert first["score"] == second["score"]
    def hash_pose(row):
        return hashlib.sha256(Path(row["pose"]).read_bytes()).hexdigest()

    assert hash_pose(first) == hash_pose(second)
    assert "ATOM" in Path(first["pose"]).read_text()


@pytest.mark.integration
def test_real_multiple_conformers_and_poses(tmp_path):
    if os.environ.get("ALD_RUN_DOCKING_TESTS") != "1":
        pytest.skip("Set ALD_RUN_DOCKING_TESTS=1 to run native molecular docking")
    pytest.importorskip("vina")
    pytest.importorskip("meeko")
    configuration = {
        "backend": "vina", "receptor": str(ROOT / "examples/data/receptor.pdbqt"),
        "reference_ligand": str(ROOT / "inputs/reference.pdb"), "box_size": [20, 20, 20],
        "exhaustiveness": 1, "n_poses": 3, "conformers": 2, "timeout": 120,
    }
    record = DockingOracle(configuration, seed=42, cpu=1).evaluate(
        "Z942299110", "O=C(c1n[nH]c(=O)c2ccccc12)N1CCC(c2ccccc2)C1", tmp_path)
    assert record["status"] == "ok", record
    assert len(record["conformers"]) == 2
    assert record["score"] == min(item["score"] for item in record["conformers"])
    assert len({item["seed"] for item in record["conformers"]}) == 2
    for item in record["conformers"]:
        pose = Path(item["pose"]).read_text()
        assert 1 <= pose.count("MODEL ") <= 3
