#!/usr/bin/env python3
"""Prepare a receptor with strict Meeko typing and an auditable provenance file.

Missing-atom rebuilding is opt-in. Existing heavy-atom coordinates are retained;
PDBFixer reconstructs missing sidechains/terminal atoms but never missing residues.
No chains, waters, cofactors or unmatched residues are silently deleted.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from active_learning_docking.provenance import validate_new_paths


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Protein PDB; strict typing by default")
    parser.add_argument("--output", required=True, type=Path, help="Prepared receptor PDBQT")
    parser.add_argument("--repair-missing-atoms", action="store_true",
                        help="Explicitly rebuild missing atoms with PDBFixer and regenerate hydrogens")
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    args.input, args.output = args.input.expanduser(), args.output.expanduser()
    if not 1 <= args.cpu < 10:
        parser.error("--cpu must be between 1 and 9")
    if not 0 <= args.ph <= 14 or not 1 <= args.seed < 2**31 or args.timeout <= 0:
        parser.error("require pH in [0, 14], seed in [1, 2**31), and positive timeout")
    if not args.input.is_file() or args.output.suffix.lower() != ".pdbqt":
        parser.error("input must exist and output must end in .pdbqt")
    outputs = [args.output, args.output.with_suffix(".preparation.log"), args.output.with_suffix(".provenance.json")]
    if args.repair_missing_atoms:
        outputs.append(args.output.with_suffix(".prepared.pdb"))
    try:
        validate_new_paths(outputs, inputs=[args.input])
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.input, args.output = args.input.resolve(), args.output.resolve()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENMM_CPU_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if hasattr(os, "sched_setaffinity"):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available[:args.cpu])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = args.input
    provenance = {
        "source": str(args.input), "source_sha256": sha256(args.input),
        "output": str(args.output), "seed": args.seed, "cpu": args.cpu,
        "repair_missing_atoms": args.repair_missing_atoms, "meeko": version("meeko"),
        "charge_model": "gasteiger", "deleted_residues": [], "rebuilt_residues": False,
        "command": [str(Path(sys.argv[0])), *sys.argv[1:]],
    }
    if args.repair_missing_atoms:
        import numpy as np
        from openmm import Platform
        from openmm.app import Modeller, PDBFile, element
        from pdbfixer import PDBFixer

        random.seed(args.seed)
        np.random.seed(args.seed)
        fixer = PDBFixer(filename=str(args.input), platform=Platform.getPlatformByName("Reference"))
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.delete([atom for atom in modeller.topology.atoms() if atom.element == element.hydrogen])
        fixer.topology, fixer.positions = modeller.topology, modeller.positions
        fixer.missingResidues = {}  # Do not infer/rebuild loops or absent terminal residues.
        fixer.findMissingAtoms()
        provenance["missing_atoms"] = {
            f"{res.chain.id}:{res.id}": [atom.name for atom in atoms]
            for res, atoms in fixer.missingAtoms.items()
        }
        provenance["missing_terminal_atoms"] = {
            f"{res.chain.id}:{res.id}": list(atoms) for res, atoms in fixer.missingTerminals.items()
        }
        fixer.addMissingAtoms(seed=args.seed)
        fixer.addMissingHydrogens(pH=args.ph)
        source = args.output.with_suffix(".prepared.pdb")
        with source.open("x") as stream:
            PDBFile.writeFile(fixer.topology, fixer.positions, stream, keepIds=True)
        provenance.update({"prepared_pdb": str(source), "prepared_pdb_sha256": sha256(source),
                           "pdbfixer": version("pdbfixer"), "openmm": version("openmm"),
                           "openmm_platform": "Reference", "ph": args.ph,
                           "hydrogens": "input hydrogens replaced with PDBFixer default pKa assignments"})
    command = [sys.executable, "-m", "meeko.cli.mk_prepare_receptor", "--read_pdb", str(source),
               "--write_pdbqt", str(args.output), "--charge_model", "gasteiger"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=False)
    log = args.output.with_suffix(".preparation.log")
    log.write_text(result.stdout + result.stderr)
    if result.returncode != 0 or not args.output.is_file():
        raise SystemExit(f"Strict Meeko preparation failed. Inspect {log}; no residues were silently removed.")
    provenance["output_sha256"] = sha256(args.output)
    provenance["meeko_command"] = command[1:]
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": provenance["output_sha256"]}))


if __name__ == "__main__":
    main()
