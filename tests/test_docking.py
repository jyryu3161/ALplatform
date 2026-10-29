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
    kill_session = Mock()
    monkeypatch.setattr(docking, "_kill_session", kill_session)
    with pytest.raises(type(original)) as error:
        _run_bounded(["worker"], 0.1, {})
    assert error.value is original
    killpg.assert_called_once_with(process.pid, signal.SIGKILL)
    kill_session.assert_called_once_with(process.pid)
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


# --- GNINA rescoring contract with fake Apptainer, fake SIF and fake QVina -------------------------

GNINA_VERSION = "gnina v1.3.3 master:6fe1ce2   Built Jun 30 2026."
GNINA_MODELS = ["dense_1_3", "dense_1_3_PT_KD_3", "crossdock_default2018_KD_4"]
RESCORING_FIELDS = ("docking_energy", "cnn_score", "cnn_affinity")

FAKE_MODEL_QVINA = '''#!{python}
"""Fake QVina: wrap the Meeko ligand in MODEL blocks; model m is shifted by m Angstrom along x."""
import json, sys
from pathlib import Path
here = Path(__file__).resolve().parent
args = sys.argv[1:]
ligand = Path(args[args.index("--ligand") + 1])
out = Path(args[args.index("--out") + 1])
table = json.loads((here / "energies.json").read_text())
energies = table[min(int(ligand.stem.split("_")[-1]), len(table) - 1)]
text = []
for model, energy in enumerate(energies):
    text.append(f"MODEL {{model + 1}}\\nREMARK VINA RESULT: {{energy:>9}}      0.000      0.000\\n")
    for line in ligand.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            line = line[:30] + f"{{float(line[30:38]) + model:8.3f}}" + line[38:]
        text.append(line + "\\n")
    text.append("ENDMDL\\n")
out.write_text("".join(text))
'''

FAKE_APPTAINER = '''#!{python}
"""Fake Apptainer: record arguments, emulate `--version`, `gnina --version` and a GNINA rescoring."""
import json, os, subprocess, sys, time
from pathlib import Path
here = Path(__file__).resolve().parent
behaviour = json.loads((here / "behaviour.json").read_text())
args = sys.argv[1:]
leaked = sorted(k for k in os.environ if k.startswith(("APPTAINER", "SINGULARITY")))
with (here / "calls.jsonl").open("a") as log:
    log.write(json.dumps({{"args": args, "apptainer_environment": leaked}}) + "\\n")
if args == ["--version"]:
    print("apptainer version 1.5.5-1")
    sys.exit(0)
command = args[args.index("gnina") + 1:]
if command == ["--version"]:
    print(behaviour.get("version_output", {version!r}))
    sys.exit(0)
work = Path(next(a for a in args if a.endswith(":/work"))[:-len(":/work")])
mode = behaviour.get("mode", "ok")
for prefix, special in behaviour.get("by_directory", {{}}).items():
    if work.name.startswith(prefix):
        mode = special
if mode == "sleep":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], process_group=0)
    (here / "sleep.pids").write_text(f"{{os.getpid()}} {{child.pid}}")
    time.sleep(60)
if mode == "exit":
    print("fake GNINA failure")
    sys.exit(3)
if mode == "no_output":
    sys.exit(0)
affinity = behaviour.get("affinity", "6.5974888802")
if behaviour.get("affinity_from_name"):
    affinity = f"{{4 + (sum(work.name.encode()) % 97) / 32:.10f}}"
items = [("minimizedAffinity", "-9.47133"), ("CNNscore", "0.8160941601"), ("CNNaffinity", affinity),
         ("CNN_VS", "5.3840069771"), ("CNNaffinity_variance", "0.0581536554")]
if mode == "missing":
    items = [item for item in items if item[0] != "CNNaffinity"]
elif mode in ("nan", "inf"):
    items = [(name, mode if name == "CNNaffinity" else value) for name, value in items]
elif mode == "duplicate":
    items.append(("CNNaffinity", affinity))
ligand = (work / "rescore_pose.sdf").read_text()
record = ligand[:-len("$$$$\\n")] + "".join(f">  <{{name}}>  (1) \\n{{value}}\\n\\n" for name, value in items) + "$$$$\\n"
with (work / "gnina_scored.sdf").open("x") as handle:
    handle.write(record * (2 if mode == "two_records" else 1))
print("Affinity: -9.47133 (kcal/mol)")
'''


def write_fake_gnina(directory, behaviour=None, energies=(("-7.250",),)):
    """Create a fake QVina, fake Apptainer and fake SIF; return the docking settings that use them."""
    directory.mkdir(parents=True, exist_ok=True)
    qvina, apptainer, image = directory / "fake qvina", directory / "apptainer", directory / "gnina.sif"
    qvina.write_text(FAKE_MODEL_QVINA.format(python=sys.executable))
    apptainer.write_text(FAKE_APPTAINER.format(python=sys.executable, version=GNINA_VERSION))
    for path in (qvina, apptainer):
        path.chmod(0o755)
    image.write_bytes(b"fake SIF image for tests\n")
    (directory / "energies.json").write_text(json.dumps([list(row) for row in energies]))
    (directory / "behaviour.json").write_text(json.dumps(behaviour or {}))
    return {"backend": "qvina", "executable": str(qvina), "rescoring": "gnina",
            "gnina_image": str(image), "apptainer_executable": str(apptainer)}


def gnina_calls(directory):
    """Recorded fake-Apptainer invocations, in call order."""
    path = directory / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _rescoring_oracle(tmp_path, behaviour=None, energies=(("-7.250",),), **settings):
    pytest.importorskip("meeko")
    receptor = tmp_path / "protein.pdbqt"
    receptor.write_text("ATOM      1  C   ALA A   1       1.000   2.000   3.000  1.00  0.00     0.000 C\n")
    configuration = {"receptor": str(receptor), "center": [1, 2, 3], "box_size": [10, 10, 10],
                     "exhaustiveness": 1, "n_poses": 1, "timeout": 60,
                     **write_fake_gnina(tmp_path / "fake", behaviour, energies), **settings}
    return DockingOracle(configuration, seed=42, cpu=1)


def _model_block(pose_text, number):
    """Lines of MODEL ``number`` (1-based) without the MODEL/ENDMDL records."""
    blocks = pose_text.split("ENDMDL\n")[:-1]
    return "\n".join(blocks[number - 1].splitlines()[1:]) + "\n"


def test_gnina_rescoring_success_records_negative_cnn_affinity_and_isolated_command(tmp_path, monkeypatch):
    from active_learning_docking.provenance import sha256_file
    from rdkit import Chem

    monkeypatch.setenv("APPTAINER_BIND", "/etc:/leak")
    monkeypatch.setenv("SINGULARITY_BIND", "/etc:/leak")
    oracle = _rescoring_oracle(tmp_path, energies=(("-7.250", "-6.125"),))
    fake = (tmp_path / "fake").resolve()
    assert oracle.rescoring == {
        "engine": "gnina", "image": str(fake / "gnina.sif"), "image_sha256": sha256_file(fake / "gnina.sif"),
        "apptainer": str(fake / "apptainer"), "apptainer_version": "apptainer version 1.5.5-1",
        "version": GNINA_VERSION, "cnn_model": GNINA_MODELS, "input_format": "sdf"}
    record = oracle.evaluate("Z1", "CC(N)=O", tmp_path / "output")
    assert record["status"] == "ok", record
    assert record["score"] == -6.5974888802
    assert (record["docking_energy"], record["cnn_score"], record["cnn_affinity"]) == (
        -7.25, 0.8160941601, 6.5974888802)
    directory = Path(record["pose"]).parent
    assert Path(record["pose"]).name == "rescore_pose.sdf"
    assert json.loads((directory / "result.json").read_text()) == record
    request = json.loads((directory / "request.json").read_text())
    assert (request["rescoring"], request["gnina_image"], request["apptainer_executable"]) == (
        "gnina", str(fake / "gnina.sif"), str(fake / "apptainer"))
    assert {p.name for p in directory.iterdir()} >= {"rescore_pose.pdbqt", "rescore_pose.sdf",
                                                     "gnina_scored.sdf", "gnina.log", "best_poses.pdbqt"}
    assert (directory / "rescore_pose.pdbqt").read_text() == _model_block(
        (directory / "best_poses.pdbqt").read_text(), 1)
    sdf = (directory / "rescore_pose.sdf").read_text()
    assert sdf.count("$$$$") == 1 and ">  <" not in sdf
    molecule = Chem.MolFromMolBlock(sdf, removeHs=False)
    assert Chem.MolToSmiles(Chem.RemoveHs(molecule)) == Chem.MolToSmiles(Chem.MolFromSmiles("CC(N)=O"))
    pdbqt_heavy = [[float(line[i:i + 8]) for i in (30, 38, 46)] for line in
                   (directory / "rescore_pose.pdbqt").read_text().splitlines()
                   if line.startswith("ATOM") and line.split()[-1] != "HD"]
    sdf_heavy = [list(molecule.GetConformer().GetAtomPosition(a.GetIdx())) for a in molecule.GetAtoms()
                 if a.GetAtomicNum() > 1]
    assert sorted(map(tuple, pdbqt_heavy)) == pytest.approx(sorted(map(tuple, sdf_heavy)), abs=1e-3)
    assert "Affinity" in (directory / "gnina.log").read_text()
    calls = gnina_calls(fake)
    assert [c["args"][-1] for c in calls[:2]] == ["--version", "--version"]
    assert all(c["apptainer_environment"] == [] for c in calls)
    args = calls[-1]["args"]
    image = args.index(str(fake / "gnina.sif"))
    assert args[:image] == [
        str(fake / "apptainer"), "exec", "--containall", "--cleanenv", "--no-home", "--net", "--network", "none",
        "--pwd", "/", "--env", "CUDA_VISIBLE_DEVICES=",
        "--bind", f"{(tmp_path / 'protein.pdbqt').resolve()}:/input/receptor.pdbqt:ro",
        "--bind", f"{directory}:/work"][1:]
    assert "--nv" not in args
    gnina = args[image + 1:]
    seed = compound_seed(record["seed"], "gnina", "CC(N)=O")
    assert gnina == ["gnina", "--receptor", "/input/receptor.pdbqt", "--ligand", "/work/rescore_pose.sdf",
                     "--score_only", "--cnn_scoring", "rescore", "--cnn", *GNINA_MODELS, "--no_gpu",
                     "--cpu", "1", "--seed", str(seed), "--out", "/work/gnina_scored.sdf"]


def test_gnina_version_banner_lines_are_allowed(tmp_path):
    banner = "==========\n== CUDA ==\n==========\nCUDA Version 12.6.0\n\n"
    oracle = _rescoring_oracle(tmp_path, {"version_output": banner + GNINA_VERSION})
    assert oracle.rescoring["version"] == GNINA_VERSION


@pytest.mark.parametrize("version_output", [
    "gnina v1.3.2 master:abcdef0   Built Jan 1 2026.", "gnina v1.3.33 master:6fe1ce2", "no version here",
    GNINA_VERSION + "\n" + GNINA_VERSION, "",
])
def test_gnina_version_mismatch_fails_at_oracle_creation_before_any_docking(tmp_path, version_output):
    with pytest.raises(RuntimeError, match="1.3.3"):
        _rescoring_oracle(tmp_path, {"version_output": version_output})
    assert not any("--score_only" in c["args"] for c in gnina_calls(tmp_path / "fake"))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("missing", ["gnina_image", "apptainer_executable"])
def test_missing_apptainer_or_image_fails_at_oracle_creation(tmp_path, missing):
    with pytest.raises(ValueError, match=missing):
        _rescoring_oracle(tmp_path, **{missing: str(tmp_path / "absent")})


@pytest.mark.parametrize("settings", [
    {"rescoring": "bogus"}, {"rescoring": "gnina", "gnina_image": None},
    {"rescoring": "gnina", "apptainer_executable": None}, {"rescoring": "none"},
])
def test_invalid_oracle_rescoring_settings_rejected(tmp_path, settings):
    with pytest.raises(ValueError, match="rescoring|gnina_image|apptainer_executable"):
        _rescoring_oracle(tmp_path, **settings)


@pytest.mark.parametrize("mode,message", [
    ("exit", "GNINA exited 3"), ("missing", "CNNaffinity"), ("nan", "non-finite CNNaffinity"),
    ("inf", "non-finite CNNaffinity"), ("duplicate", "CNNaffinity"), ("two_records", "exactly one"),
    ("no_output", "no scored pose"),
])
def test_gnina_failures_spend_the_attempt_without_score_or_rescoring_values(tmp_path, mode, message):
    record = _rescoring_oracle(tmp_path, {"mode": mode}).evaluate("Z1", "CCO", tmp_path / "output")
    assert record["status"] == "failed"
    assert record["score"] is None and record["pose"] is None
    assert message in record["error"], record["error"]
    assert not any(name in record for name in RESCORING_FIELDS)
    directory = next((tmp_path / "output").iterdir())
    assert json.loads((directory / "result.json").read_text()) == record
    assert (directory / "best_poses.pdbqt").is_file() and (directory / "gnina.log").is_file()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-state inspection")
def test_gnina_timeout_kills_the_whole_worker_session(tmp_path):
    oracle = _rescoring_oracle(tmp_path, {"mode": "sleep"}, timeout=10)
    marker = tmp_path / "fake" / "sleep.pids"
    start = time.monotonic()
    try:
        record = oracle.evaluate("Z1", "CCO", tmp_path / "output")
        assert time.monotonic() - start < 15
        assert record["status"] == "failed" and record["score"] is None
        assert "timed out" in record["error"]
        for pid in marker.read_text().split():
            _assert_process_gone(int(pid))
    finally:
        _kill_recorded(marker)


def _assert_process_gone(pid):
    for _ in range(200):
        try:
            if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] in {"Z", "X"}:
                return
        except (FileNotFoundError, ProcessLookupError):
            return
        time.sleep(0.01)
    pytest.fail(f"process {pid} of the timed-out worker session is still running")


def _kill_recorded(marker):
    if marker.exists():
        for pid in marker.read_text().split():
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-state inspection")
def test_timeout_kills_other_process_groups_of_the_worker_session(tmp_path):
    marker = tmp_path / "group.pid"
    script = (
        "import subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], process_group=0); "
        f"Path({str(marker)!r}).write_text(str(child.pid)); time.sleep(30)"
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            _run_bounded([sys.executable, "-c", script], 1.0, os.environ.copy())
        _assert_process_gone(int(marker.read_text()))
    finally:
        _kill_recorded(marker)


def test_rescoring_selects_recorded_energy_model_of_best_conformer_first_on_ties(tmp_path):
    energies = (("-5.000",), ("-6.000", "-7.500", "-7.500"))
    record = _rescoring_oracle(tmp_path, energies=energies, conformers=2).evaluate(
        "Z1", "CCO", tmp_path / "output")
    assert record["status"] == "ok", record
    assert record["best_conformer"] == 1 and record["docking_energy"] == -7.5
    directory = Path(record["pose"]).parent
    expected = _model_block((directory / "poses_001.pdbqt").read_text(), 2)
    assert (directory / "rescore_pose.pdbqt").read_text() == expected
    assert (directory / "best_poses.pdbqt").read_bytes() == (directory / "poses_001.pdbqt").read_bytes()


def test_select_model_matches_printed_engine_precision_and_first_tie():
    def model(number, energy, x):
        return (f"MODEL {number}\nREMARK VINA RESULT: {energy:>9}      0.000      0.000\n"
                f"ATOM      1  C   UNL     1    {x:8.3f}   0.000   0.000  1.00  0.00     0.000 C \nENDMDL\n")

    text = model(1, "-8.998", 1.0) + model(2, "-7.500", 2.0) + model(3, "-7.500", 3.0)
    assert "   1.000" in docking._select_model(text, -8.99802)
    assert "   2.000" in docking._select_model(text, -7.5)
    assert not docking._select_model(text, -7.5).startswith(("MODEL", "ENDMDL"))
    assert "   1.000" in docking._select_model(model(1, "-9.0", 1.0), -9.0)
    for bad_text, energy in [(text, -8.5), ("REMARK VINA RESULT: -7.5 0 0\nATOM\n", -7.5),
                             ("MODEL 1\nATOM\nENDMDL\n", -7.5), ("MODEL 1\nREMARK VINA RESULT: -7.5 0 0\n", -7.5)]:
        with pytest.raises(ValueError):
            docking._select_model(bad_text, energy)


@pytest.mark.parametrize("target", ["rescore_pose.pdbqt", "rescore_pose.sdf", "gnina_scored.sdf", "gnina.log"])
@pytest.mark.parametrize("is_link", [False, True])
def test_rescoring_outputs_are_checked_before_preparation(tmp_path, monkeypatch, target, is_link):
    path = tmp_path / target
    if is_link:
        path.symlink_to(tmp_path / "absent")
    else:
        path.write_bytes(b"original evidence")
    monkeypatch.setattr(docking, "prepare_ligands", lambda *args: pytest.fail("preflight must run first"))
    with pytest.raises(FileExistsError):
        docking._dock({"directory": str(tmp_path), "backend": "qvina", "conformers": 1, "rescoring": "gnina"})
    assert list(tmp_path.iterdir()) == [path]
    assert path.is_symlink() if is_link else path.read_bytes() == b"original evidence"


def test_rescoring_off_keeps_request_result_and_files_unchanged(tmp_path):
    oracle = _fake_oracle(tmp_path)
    assert oracle.rescoring is None
    record = oracle.evaluate("Z1", "CCO", tmp_path / "output")
    assert list(record) == ["cid", "score", "status", "error", "pose", "seed", "best_conformer", "conformers"]
    directory = Path(record["pose"]).parent
    assert list(json.loads((directory / "request.json").read_text())) == [
        "backend", "receptor", "center", "box_size", "executable", "cpu", "exhaustiveness", "n_poses",
        "conformers", "cid", "smiles", "seed", "directory"]
    assert {p.name for p in directory.iterdir()} == {
        "request.json", "result.json", "worker.log", "conformer_000.pdbqt", "poses_000.pdbqt",
        "poses_000.args.json", "engine_000.log", "best_poses.pdbqt"}
