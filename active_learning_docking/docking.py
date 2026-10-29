"""Seeded ligand preparation and isolated, resource-bounded docking oracles.

Both backends use Meeko PDBQT preparation and choose the minimum Vina energy
over all generated conformers. Scores are docking energies in kcal/mol, not
experimentally measured affinities. No reference-pose RMSD filtering is applied.

With ``rescoring: gnina`` the worker also rescores the single best pose, as a
bond-order SDF, with the pinned GNINA 1.3.3 default CNN ensemble (score-only, CPU,
no network) inside a rootless Apptainer image. The score is then -CNNaffinity
(pK units; lower is better), and the docking energy is kept as ``docking_energy``.
"""

from __future__ import annotations

from contextlib import suppress
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from typing import Any


def compound_seed(seed: int, cid: str, smiles: str) -> int:
    """A stable, nonzero 31-bit seed independent of evaluation order."""
    payload = json.dumps([int(seed), str(cid), smiles], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big") % (2**31 - 2) + 1


REFERENCE_FORMATS = (".pdb", ".pdbqt", ".sdf", ".mol", ".mol2")


def _pdb_heavy_atoms(path: Path) -> list[list[float]]:
    coords = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        element = line[76:78].strip().upper() if path.suffix.lower() == ".pdb" else line.split()[-1].upper()
        if element in {"H", "HD", "HS", "D"} or (not element and line[12:16].strip().lstrip("0123456789").startswith("H")):
            continue
        coords.append([float(line[start:start + 8]) for start in (30, 38, 46)])
    return coords


def _rdkit_heavy_atoms(path: Path) -> list[list[float]]:
    """Heavy-atom coordinates of the first 3D record in an SDF/MOL/MOL2 file."""
    from rdkit import Chem

    try:
        if path.suffix.lower() == ".mol2":
            molecule = Chem.MolFromMol2File(str(path), sanitize=False, removeHs=False)
        else:
            molecule = next(iter(Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)), None)
    except OSError as exc:
        raise ValueError(f"reference ligand could not be parsed: {exc}") from exc
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError("reference ligand could not be parsed or has no 3D coordinates")
    conformer = molecule.GetConformer()
    if not conformer.Is3D():
        raise ValueError("reference ligand must contain 3D coordinates")
    return [list(conformer.GetAtomPosition(atom.GetIdx()))
            for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]


def reference_center(filename: str | Path) -> list[float]:
    """Geometric heavy-atom centroid, in Angstrom, of a PDB/PDBQT/SDF/MOL/MOL2 ligand.

    Hydrogens are excluded so the center does not depend on the protonation
    model of the reference. Multi-record SDF files use their first record.
    """
    path = Path(filename)
    if path.suffix.lower() not in REFERENCE_FORMATS:
        raise ValueError(f"reference_ligand must be one of {REFERENCE_FORMATS}; otherwise set center explicitly")
    if path.suffix.lower() in {".pdb", ".pdbqt"}:
        coords = _pdb_heavy_atoms(path)
    else:
        coords = _rdkit_heavy_atoms(path)
    if not coords:
        raise ValueError("reference ligand contains no heavy-atom coordinates")
    if not all(math.isfinite(x) for row in coords for x in row):
        raise ValueError("reference ligand contains non-finite coordinates")
    return [sum(row[i] for row in coords) / len(coords) for i in range(3)]


def _vector(value: Any, name: str, positive: bool = False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")
    result = [float(x) for x in value]
    if not all(math.isfinite(x) and (not positive or x > 0) for x in result):
        raise ValueError(f"{name} must contain finite {'positive ' if positive else ''}values")
    return result


GNINA_VERSION_PREFIX = "gnina v1.3.3 "
# GNINA 1.3.3 default CNN ensemble, named explicitly and in order so that an image
# upgrade cannot silently change the score definition (float32 averaging is order dependent).
GNINA_CNN_MODELS = ("dense_1_3", "dense_1_3_PT_KD_3", "crossdock_default2018_KD_4")
RESCORING_FILES = ("rescore_pose.pdbqt", "rescore_pose.sdf", "gnina_scored.sdf", "gnina.log")
RESCORING_VALUES = ("docking_energy", "cnn_score", "cnn_affinity")


def _kill_session(session: int) -> None:
    """SIGKILL every live process left in a POSIX session (Linux ``/proc`` scan).

    Rootless Apptainer runs its SIF mount helper (``squashfuse_ll``) in a separate
    process group of the worker's session, so killing the worker process group
    alone leaves it running. Errors are ignored: cleanup must never replace the
    exception that triggered it. Platforms without ``/proc`` are skipped.

    Args:
        session: Session ID, which is the worker PID for a ``start_new_session`` child.
    """
    proc = Path("/proc")
    if not proc.is_dir():
        return
    for _ in range(20):  # Repeat while processes forked during the previous sweep remain.
        found = False
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                # Fields after the parenthesized command: state, ppid, pgrp, session, ...
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[3]) != session or fields[0] in {"Z", "X"}:
                    continue
                os.kill(int(entry.name), signal.SIGKILL)
            except (OSError, ValueError, IndexError):
                continue  # The process exited meanwhile or is not ours to inspect.
            found = True
        if not found:
            return


def _run_bounded(command: list[str], timeout: float, environment: dict[str, str]) -> subprocess.CompletedProcess:
    """Run ``command`` in a new session; on timeout or interruption SIGKILL the whole session.

    The worker's process group is killed first (engine children included), then
    any process remaining in the worker's session, such as Apptainer helpers that
    run in their own process group.
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, env=environment, start_new_session=True)
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except BaseException:
        try:
            if os.name == "posix":
                # The worker may have exited while its engine still holds stdout.
                os.killpg(process.pid, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
        except OSError:
            pass  # The process group may already have disappeared.
        if os.name == "posix":
            _kill_session(process.pid)
        try:
            process.communicate(timeout=1.0)
        except BaseException:
            # Cleanup is bounded and must never replace the original exception.
            with suppress(BaseException):
                if process.stdout is not None:
                    process.stdout.close()
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, None)


def _worker_environment() -> dict[str, str]:
    """Environment for docking children: one thread per numerical library and no CUDA device."""
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[name] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""  # CPU docking never initializes CUDA.
    return environment


def _apptainer_environment(environment: dict[str, str]) -> dict[str, str]:
    """Copy ``environment`` without APPTAINER_*/SINGULARITY_* variables.

    Apptainer reads such variables as options (for example ``APPTAINER_BIND``),
    which could widen the container beyond the receptor and compound directory.
    """
    return {name: value for name, value in environment.items()
            if not name.startswith(("APPTAINER", "SINGULARITY"))}


def _apptainer_exec(apptainer: str, image: str, binds: tuple[str, ...] = ()) -> list[str]:
    """Return the ``apptainer exec`` prefix used for every GNINA call.

    No GPU (no ``--nv``, empty ``CUDA_VISIBLE_DEVICES``), no network namespace
    access, no host environment or home directory, and only the given binds.
    """
    command = [apptainer, "exec", "--containall", "--cleanenv", "--no-home", "--net", "--network", "none",
               "--pwd", "/", "--env", "CUDA_VISIBLE_DEVICES="]
    for bind in binds:
        command.extend(["--bind", bind])
    return command + [image]


def _gnina_provenance(apptainer: Any, image: Any, timeout: float) -> dict[str, Any]:
    """Check Apptainer, the SIF image and the pinned GNINA version before any docking budget is spent.

    Args:
        apptainer: Apptainer executable path (never looked up on PATH).
        image: GNINA Apptainer SIF image path.
        timeout: Seconds allowed for each version query.

    Returns:
        The manifest ``rescoring`` record, including the SIF SHA-256.

    Raises:
        ValueError: A path is missing or unusable.
        RuntimeError: Apptainer fails or the image does not report exactly one GNINA 1.3.3 version line.
    """
    from .provenance import sha256_file

    if not apptainer or not image:
        raise ValueError("docking.rescoring: gnina requires docking.gnina_image and docking.apptainer_executable")
    apptainer_path = Path(apptainer).expanduser().resolve()
    image_path = Path(image).expanduser().resolve()
    if not apptainer_path.is_file() or not os.access(apptainer_path, os.X_OK):
        raise ValueError(f"docking.apptainer_executable is missing or not executable: {apptainer_path}")
    if not image_path.is_file():
        raise ValueError(f"docking.gnina_image is not an existing SIF file: {image_path}")
    environment = _apptainer_environment(_worker_environment())
    outputs = []
    for command in ([str(apptainer_path), "--version"],
                    _apptainer_exec(str(apptainer_path), str(image_path)) + ["gnina", "--version"]):
        try:
            completed = _run_bounded(command, timeout, environment)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{' '.join(command)} timed out after {timeout:g} seconds") from exc
        if completed.returncode != 0:
            raise RuntimeError(f"{' '.join(command)} exited {completed.returncode}: {completed.stdout[-1000:]}")
        outputs.append(completed.stdout)
    versions = [line for line in outputs[1].splitlines() if line.startswith("gnina v")]
    if len(versions) != 1 or not versions[0].startswith(GNINA_VERSION_PREFIX):
        raise RuntimeError(f"GNINA image must report exactly one '{GNINA_VERSION_PREFIX.strip()}' version line; "
                           f"found {versions}")
    return {"engine": "gnina", "image": str(image_path), "image_sha256": sha256_file(image_path),
            "apptainer": str(apptainer_path), "apptainer_version": outputs[0].strip(),
            "version": versions[0], "cnn_model": list(GNINA_CNN_MODELS), "input_format": "sdf"}


class DockingOracle:
    """Evaluate compounds using Vina's official Python package or a QVina binary.

    Preparation and docking run in a child process so ``timeout`` applies to the
    complete evaluation, including native engine calls and all conformers. With
    ``rescoring: gnina`` the same child also rescores the best pose with GNINA, and
    ``self.rescoring`` holds the manifest record checked here before any docking.
    """

    def __init__(self, config: Any, seed: int = 42, cpu: int = 1):
        def get(name, default=None):
            return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)

        if not isinstance(cpu, int) or isinstance(cpu, bool) or not 1 <= cpu < 10:
            raise ValueError("docking cpu must be an integer between 1 and 9")
        self.seed = int(seed)
        self.cpu = cpu
        backend = get("backend", "vina")
        if backend not in {"vina", "qvina"}:
            raise ValueError("docking backend must be 'vina' or 'qvina'")
        receptor = Path(get("receptor") or "").expanduser().resolve()
        if not receptor.is_file() or receptor.suffix.lower() != ".pdbqt":
            raise ValueError("docking receptor must be an existing, prepared PDBQT file")
        if not any(x.startswith(("ATOM  ", "HETATM")) for x in receptor.read_text().splitlines()):
            raise ValueError("docking receptor PDBQT contains no atoms")
        center = get("center")
        if center is None:
            if not get("reference_ligand"):
                raise ValueError("set docking.center or docking.reference_ligand")
            center = reference_center(get("reference_ligand"))
        executable = get("executable", "qvina2.1")
        if backend == "qvina":
            executable = shutil.which(str(executable))
            if executable is None:
                raise ValueError("QVina executable is missing or not executable")
            executable = str(Path(executable).resolve())
        for dependency in ("meeko", "rdkit") + (("vina",) if backend == "vina" else ()):
            if importlib.util.find_spec(dependency) is None:
                raise ImportError(f"Install the docking extra: missing {dependency}")
        self.parameters = {
            "backend": backend, "receptor": str(receptor), "center": _vector(center, "center"),
            "box_size": _vector(get("box_size", [20, 20, 20]), "box_size", positive=True),
            "executable": executable, "cpu": cpu,
        }
        for name, default in (("exhaustiveness", 8), ("n_poses", 3), ("conformers", 1)):
            value = get(name, default)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"docking.{name} must be a positive integer")
            self.parameters[name] = value
        self.timeout = float(get("timeout", 300.0))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("docking.timeout must be finite and positive")
        rescoring = get("rescoring", "none")
        self.rescoring = None
        if rescoring == "gnina":
            if any(character in str(receptor) for character in ":,"):
                raise ValueError("Apptainer cannot bind a receptor path containing ':' or ','")
            self.rescoring = _gnina_provenance(get("apptainer_executable"), get("gnina_image"), self.timeout)
            self.parameters.update(rescoring="gnina", gnina_image=self.rescoring["image"],
                                   apptainer_executable=self.rescoring["apptainer"])
        elif rescoring != "none":
            raise ValueError("docking.rescoring must be none or gnina")
        elif get("gnina_image") is not None or get("apptainer_executable") is not None:
            raise ValueError("docking.gnina_image and docking.apptainer_executable require docking.rescoring: gnina")

    def evaluate(self, cid: str, smiles: str, output_dir: str | Path) -> dict[str, Any]:
        """Return a score/failure; raise FileExistsError before reusing any evidence."""
        cid = str(cid)
        seed = compound_seed(self.seed, cid, smiles)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", cid).strip("._")[:48] or "compound"
        digest = hashlib.sha256(json.dumps([cid, smiles]).encode()).hexdigest()[:16]
        directory = Path(output_dir).resolve() / f"{slug}_{digest}"
        if directory.is_symlink() or (directory.exists() and (
            not directory.is_dir() or any(directory.iterdir())
        )):
            raise FileExistsError(f"compound output directory is not empty or is a link: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        request = dict(self.parameters, cid=cid, smiles=smiles, seed=seed, directory=str(directory))
        request_path, result_path = directory / "request.json", directory / "result.json"
        with request_path.open("x") as handle:
            handle.write(json.dumps(request, indent=2, allow_nan=False) + "\n")
        record = {"cid": cid, "score": None, "status": "failed", "error": None, "pose": None, "seed": seed}
        try:
            completed = _run_bounded([sys.executable, str(Path(__file__).resolve()), "--worker",
                                      str(request_path), str(result_path)], self.timeout, _worker_environment())
            (directory / "worker.log").write_text(completed.stdout)
            if completed.returncode != 0 or not result_path.is_file():
                raise RuntimeError(f"docking worker exited {completed.returncode}: {completed.stdout[-1500:]}")
            result = json.loads(result_path.read_text())
            if result["status"] == "ok":
                score = float(result["score"])
                if not math.isfinite(score) or not Path(result["pose"]).is_file():
                    raise RuntimeError("docking worker returned an invalid score or missing pose")
                if self.parameters.get("rescoring") == "gnina" and not all(
                        math.isfinite(float(result[name])) for name in RESCORING_VALUES):
                    raise RuntimeError("docking worker returned non-finite rescoring values")
            record.update(result)
        except subprocess.TimeoutExpired:
            record["error"] = f"docking evaluation timed out after {self.timeout:g} seconds"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        result_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        return record


def prepare_ligands(smiles: str, seed: int, conformers: int) -> list[str]:
    """ETKDGv3 embedding, MMFF94/UFF relaxation, and Meeko Gasteiger typing.

    SMILES protonation, tautomerism and specified stereochemistry are retained;
    this routine does not enumerate alternative chemical states or strip salts.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("invalid or empty SMILES")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("disconnected SMILES: supply one explicitly selected ligand chemical state")
    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(seed)
    parameters.numThreads = 1
    ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=conformers, params=parameters))
    if len(ids) != conformers:
        raise ValueError(f"3D embedding produced {len(ids)} of {conformers} requested conformers")
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        optimized = AllChem.MMFFOptimizeMoleculeConfs(molecule, numThreads=1, maxIters=1000)
    elif AllChem.UFFHasAllMoleculeParams(molecule):
        optimized = AllChem.UFFOptimizeMoleculeConfs(molecule, numThreads=1, maxIters=1000)
    else:
        raise ValueError("neither MMFF94 nor UFF covers all ligand atoms")
    if any(status < 0 or not math.isfinite(energy) for status, energy in optimized):
        raise ValueError("ligand force-field relaxation failed")
    prepared = []
    for conformer_id in ids:
        setups = MoleculePreparation().prepare(molecule, conformer_id=conformer_id)
        if len(setups) != 1:
            raise ValueError("ligand preparation produced an unsupported number of chemical setups")
        pdbqt, success, error = PDBQTWriterLegacy.write_string(setups[0])
        if not success:
            raise ValueError(f"Meeko preparation failed: {error}")
        prepared.append(pdbqt)
    return prepared


def _dock(request: dict[str, Any]) -> dict[str, Any]:
    directory = Path(request["directory"])
    destination = directory / "best_poses.pdbqt"
    targets = [destination]
    for index in range(request["conformers"]):
        targets.extend([directory / f"conformer_{index:03d}.pdbqt", directory / f"poses_{index:03d}.pdbqt"])
        if request["backend"] == "qvina":
            targets.append(directory / f"engine_{index:03d}.log")
    if request.get("rescoring") == "gnina":
        targets.extend(directory / name for name in RESCORING_FILES)
    for target in targets:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"docking output already exists: {target}")
    ligands = prepare_ligands(request["smiles"], request["seed"], request["conformers"])
    outcomes = []
    for index, pdbqt in enumerate(ligands):
        ligand_path = directory / f"conformer_{index:03d}.pdbqt"
        pose_path = directory / f"poses_{index:03d}.pdbqt"
        with ligand_path.open("x") as handle:
            handle.write(pdbqt)
        dock_seed = compound_seed(request["seed"], str(index), request["smiles"])
        if request["backend"] == "vina":
            from vina import Vina

            engine = Vina(sf_name="vina", cpu=request["cpu"], seed=dock_seed, verbosity=0)
            engine.set_receptor(request["receptor"])
            engine.set_ligand_from_string(pdbqt)
            engine.compute_vina_maps(center=request["center"], box_size=request["box_size"])
            engine.dock(exhaustiveness=request["exhaustiveness"], n_poses=request["n_poses"])
            energies = engine.energies(n_poses=request["n_poses"])
            if len(energies) == 0:
                raise RuntimeError("Vina produced no docking poses")
            score = float(energies[0][0])
            engine.write_poses(str(pose_path), n_poses=request["n_poses"], overwrite=False)
        else:
            command = [request["executable"], "--receptor", request["receptor"], "--ligand", str(ligand_path),
                       "--out", str(pose_path), "--cpu", str(request["cpu"]), "--seed", str(dock_seed),
                       "--exhaustiveness", str(request["exhaustiveness"]), "--num_modes", str(request["n_poses"])]
            for axis, center, size in zip("xyz", request["center"], request["box_size"]):
                command.extend([f"--center_{axis}", str(center), f"--size_{axis}", str(size)])
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            (directory / f"engine_{index:03d}.log").write_text(completed.stdout + completed.stderr)
            if completed.returncode != 0 or not pose_path.is_file():
                raise RuntimeError(f"QVina exited {completed.returncode}: {completed.stderr[-1000:]}")
            scores = [float(line.split()[3]) for line in pose_path.read_text().splitlines()
                      if line.startswith("REMARK VINA RESULT:")]
            if not scores:
                raise RuntimeError("QVina pose has no VINA RESULT energy")
            if not all(math.isfinite(x) for x in scores):
                raise RuntimeError("QVina returned a non-finite docking score")
            score = min(scores)
        if not math.isfinite(score):
            raise RuntimeError("engine returned a non-finite docking score")
        outcomes.append({"conformer": index, "score": score, "seed": dock_seed, "pose": str(pose_path)})
    best = min(outcomes, key=lambda item: (item["score"], item["conformer"]))
    with Path(best["pose"]).open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target)
    result = {"status": "ok", "score": best["score"], "pose": str(destination), "error": None,
              "best_conformer": best["conformer"], "conformers": outcomes}
    if request.get("rescoring") == "gnina":
        cnn_score, cnn_affinity, pose = _rescore_with_gnina(request, Path(best["pose"]), best["score"])
        result.update(score=-cnn_affinity, pose=str(pose), docking_energy=best["score"],
                      cnn_score=cnn_score, cnn_affinity=cnn_affinity)
    return result


def _select_model(text: str, energy: float) -> str:
    """Return the first MODEL block of a pose file whose VINA RESULT energy equals ``energy``.

    Engines print energies with fixed decimals (Vina three, QVina one), so the
    recorded energy is compared after rounding to the printed precision. QVina
    scores are parsed from these same lines and therefore match exactly. The
    returned text excludes the MODEL/ENDMDL records.

    Raises:
        ValueError: Malformed MODEL records or no MODEL with the recorded energy.
    """
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("MODEL"):
            if current is not None:
                raise ValueError("pose file has a MODEL record inside another MODEL")
            current = []
        elif line.startswith("ENDMDL"):
            if current is None:
                raise ValueError("pose file has ENDMDL without MODEL")
            blocks.append(current)
            current = None
        elif current is not None:
            current.append(line)
    if current is not None or not blocks:
        raise ValueError("pose file has no complete MODEL records")
    for lines in blocks:
        remarks = [line.split() for line in lines if line.startswith("REMARK VINA RESULT:")]
        if len(remarks) != 1 or len(remarks[0]) < 4:
            raise ValueError("each pose MODEL must contain exactly one VINA RESULT energy")
        printed = remarks[0][3]
        decimals = len(printed.partition(".")[2])
        if float(printed) == float(f"{energy:.{decimals}f}"):
            return "\n".join(lines) + "\n"
    raise ValueError(f"no pose MODEL has the recorded docking energy {energy!r}")


def _pose_to_sdf(block: str, smiles: str) -> str:
    """Convert one Meeko PDBQT pose to a single-record SDF with bond orders from ``smiles``.

    Heavy-atom and polar-hydrogen coordinates are kept from the PDBQT; RDKit adds
    the non-polar hydrogens. The result must be one molecule with one conformer
    whose canonical heavy-atom SMILES equals that of ``smiles``.
    """
    from meeko import PDBQTMolecule, RDKitMolCreate
    from rdkit import Chem

    molecules = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(block, is_dlg=False, skip_typing=True))
    if len(molecules) != 1 or molecules[0] is None:
        raise ValueError(f"pose conversion produced {len(molecules)} molecules instead of one")
    molecule = molecules[0]
    if molecule.GetNumConformers() != 1:
        raise ValueError(f"pose conversion produced {molecule.GetNumConformers()} conformers instead of one")
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    observed = Chem.MolToSmiles(Chem.RemoveHs(molecule))
    if observed != expected:
        raise ValueError(f"converted pose {observed} does not match the input SMILES {expected}")
    return Chem.MolToMolBlock(molecule) + "$$$$\n"


def _gnina_values(text: str) -> dict[str, float]:
    """Read CNNscore and CNNaffinity from a GNINA output SDF holding exactly one record.

    GNINA omits an item whose value is exactly zero, so a missing item is an error.

    Raises:
        RuntimeError: Not exactly one record, not exactly one item per name, or a non-finite value.
        ValueError: An item value is not a number.
    """
    lines = text.splitlines()
    ends = [index for index, line in enumerate(lines) if line.strip() == "$$$$"]
    if len(ends) != 1 or any(line.strip() for line in lines[ends[0] + 1:]):
        raise RuntimeError("GNINA output must contain exactly one SDF record")
    values = {}
    for name in ("CNNscore", "CNNaffinity"):
        headers = [index for index, line in enumerate(lines) if line.startswith(">") and f"<{name}>" in line]
        if len(headers) != 1 or headers[0] + 1 >= ends[0]:
            raise RuntimeError(f"GNINA output must contain exactly one {name} item; found {len(headers)}")
        value = float(lines[headers[0] + 1])
        if not math.isfinite(value):
            raise RuntimeError(f"GNINA returned a non-finite {name}")
        values[name] = value
    return values


def _rescore_with_gnina(request: dict[str, Any], poses: Path, energy: float) -> tuple[float, float, Path]:
    """Rescore the recorded best pose with GNINA in the worker (CPU, score-only, isolated).

    Writes ``rescore_pose.pdbqt`` (the MODEL with the recorded energy),
    ``rescore_pose.sdf`` (GNINA input), and ``gnina.log``; GNINA writes
    ``gnina_scored.sdf``. Every file is created once and never overwritten.

    Returns:
        ``(CNNscore, CNNaffinity, path of the scored SDF input pose)``.
    """
    directory = Path(request["directory"])
    if any(character in str(directory) for character in ":,"):
        raise ValueError("Apptainer cannot bind a compound directory containing ':' or ','")
    block = _select_model(poses.read_text(), energy)
    with (directory / "rescore_pose.pdbqt").open("x") as handle:
        handle.write(block)
    sdf = directory / "rescore_pose.sdf"
    with sdf.open("x") as handle:
        handle.write(_pose_to_sdf(block, request["smiles"]))
    scored = directory / "gnina_scored.sdf"
    if scored.exists() or scored.is_symlink():
        raise FileExistsError(f"GNINA output already exists: {scored}")
    binds = (f"{request['receptor']}:/input/receptor.pdbqt:ro", f"{directory}:/work")
    command = _apptainer_exec(request["apptainer_executable"], request["gnina_image"], binds) + [
        "gnina", "--receptor", "/input/receptor.pdbqt", "--ligand", "/work/rescore_pose.sdf",
        "--score_only", "--cnn_scoring", "rescore", "--cnn", *GNINA_CNN_MODELS, "--no_gpu",
        "--cpu", str(request["cpu"]), "--seed", str(compound_seed(request["seed"], "gnina", request["smiles"])),
        "--out", "/work/gnina_scored.sdf",
    ]
    # Same session as the worker, so a timeout also kills Apptainer and its mount helper.
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
                               env=_apptainer_environment(dict(os.environ)))
    with (directory / "gnina.log").open("x") as handle:
        handle.write(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"GNINA exited {completed.returncode}: {completed.stdout[-1000:]}")
    if not scored.is_file():
        raise RuntimeError("GNINA wrote no scored pose")
    values = _gnina_values(scored.read_text())
    return values["CNNscore"], values["CNNaffinity"], sdf


def _worker(request_path: str, result_path: str) -> None:
    if Path(result_path).exists() or Path(result_path).is_symlink():
        raise FileExistsError(f"worker result already exists: {result_path}")
    request = json.loads(Path(request_path).read_text())
    try:
        result = _dock(request)
    except Exception as exc:
        result = {"status": "failed", "score": None, "pose": None, "error": f"{type(exc).__name__}: {exc}"}
    with Path(result_path).open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("Internal worker usage: python -m active_learning_docking.docking --worker REQUEST RESULT")
    _worker(sys.argv[2], sys.argv[3])
