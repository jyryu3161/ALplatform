"""Regenerate a deliberately synthetic replay fixture; no docking is performed.

Run from the repository root: ``python examples/make_demo.py``. The score is a
fixed, transparent molecular-descriptor expression solely for software testing.
It has no physical energy units and must not be interpreted as binding affinity.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

# The generator is single-process and does not require a GPU.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from rdkit import Chem, __version__ as rdkit_version
from rdkit.Chem import Crippen, Descriptors, Lipinski


SMILES = [
    "CCO", "CCN", "CCCO", "CCCN", "CC(C)O", "CC(C)N", "CCCC", "CCOC",
    "CC(=O)O", "CC(=O)N", "CCC(=O)O", "CCC(=O)N", "CCS", "CC(=O)OC", "CCN(C)C", "CCOCC",
    "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "Oc1ccccc1", "Nc1ccccc1", "COc1ccccc1",
    "Fc1ccccc1", "Clc1ccccc1", "CC(=O)c1ccccc1", "O=C(O)c1ccccc1", "NC(=O)c1ccccc1",
    "c1ccncc1", "Cc1ccncc1", "Oc1ccncc1", "Nc1ccncc1", "c1cncnc1", "c1ccoc1", "Cc1ccoc1",
    "c1ccsc1", "Cc1ccsc1", "c1ncc[nH]1", "Cc1ncc[nH]1", "C1CCCCC1", "OC1CCCCC1",
    "NC1CCCCC1", "C1CCNCC1", "CN1CCCCC1", "C1COCCN1", "C1CCOC1", "C1CC1",
    "N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O",
]


def generate(output: Path) -> dict:
    """Write canonical structures and synthetic scores with exact provenance."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for index, smiles in enumerate(SMILES, 1):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid fixture structure: {smiles}")
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if canonical in seen:
            raise ValueError(f"Duplicate fixture structure: {canonical}")
        seen.add(canonical)
        score = -(0.025 * Descriptors.MolWt(molecule) + 0.4 * Crippen.MolLogP(molecule)
                  + 0.3 * Lipinski.NumHAcceptors(molecule) - 0.2 * Lipinski.NumHDonors(molecule))
        rows.append({"Catalog ID": f"SYN{index:03d}", "smiles": canonical,
                     "synthetic_score": f"{score:.8f}"})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Catalog ID", "smiles", "synthetic_score"])
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "kind": "synthetic_software_test_fixture",
        "n_compounds": len(rows),
        "score_column": "synthetic_score",
        "score_units": "arbitrary; not docking energy or measured affinity",
        "formula": "-(0.025*MolWt + 0.4*MolLogP + 0.3*NumHAcceptors - 0.2*NumHDonors)",
        "precision": "8 digits after decimal point",
        "rdkit_version": rdkit_version,
        "generator": "examples/make_demo.py",
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "intended_use": "Deterministic integration smoke tests only; no scientific performance claims.",
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("replay.csv"))
    print(json.dumps(generate(parser.parse_args().output), indent=2))
