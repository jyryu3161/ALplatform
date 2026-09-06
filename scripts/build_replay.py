#!/usr/bin/env python3
"""Create an exhaustive docking replay fixture from a fixed input prefix.

The first N canonical representatives (smallest valid ID per structure) are
evaluated in source-record order. Failed molecules are recorded without filling their labels. This
small fixture demonstrates computational reproducibility, not biological potency
or prospective screening performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time

from active_learning_docking.provenance import validate_new_paths


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable(path):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        return str(resolved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("inputs/candidates.csv"))
    parser.add_argument("--receptor", type=Path, default=Path("examples/data/receptor.pdbqt"))
    parser.add_argument("--reference", type=Path, default=Path("inputs/reference.pdb"))
    parser.add_argument("--output", type=Path, default=Path("examples/data/docking_scores.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/reference_docking"))
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--exhaustiveness", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--box-size", nargs=3, type=float, default=[22., 22., 22.])
    args = parser.parse_args()
    for name in ("input", "receptor", "reference", "output", "artifacts"):
        setattr(args, name, getattr(args, name).expanduser())
    if not 1 <= args.cpu < 10 or args.count < 1:
        parser.error("require cpu between 1 and 9 and count >= 1")
    if not args.input.is_file():
        parser.error("--input must be a single existing CSV file")
    outputs = [args.output, args.output.with_suffix(".provenance.json")]
    outputs += [args.artifacts / name for name in ("input_rejections.csv", "records.json", "failures.json")]
    try:
        validate_new_paths(outputs, inputs=[args.input, args.receptor, args.reference],
                           directories=[args.artifacts, args.artifacts / "poses"])
        poses = (args.artifacts / "poses").resolve()
        if any(path.resolve().is_relative_to(poses) for path in outputs[:2]):
            raise ValueError("output/provenance collides with the docking poses directory")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.input, args.receptor, args.reference = [path.resolve() for path in (args.input, args.receptor, args.reference)]
    args.output, args.artifacts = args.output.resolve(), args.artifacts.resolve()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if hasattr(os, "sched_setaffinity"):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available[:args.cpu])

    from active_learning_docking.data import load_library
    from active_learning_docking import docking

    library, rejected = load_library(args.input, score_column=None, invalid_policy="skip", deduplicate=True)
    if len(library) < args.count:
        parser.error(f"only {len(library)} accepted molecules, fewer than --count={args.count}")
    library = library.sort_values(["source_file", "source_row"], kind="stable").iloc[:args.count].copy()
    args.artifacts.mkdir(parents=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rejected.to_csv(args.artifacts / "input_rejections.csv", index=False)
    configuration = {
        "backend": "vina", "receptor": str(args.receptor), "reference_ligand": str(args.reference),
        "box_size": args.box_size, "exhaustiveness": args.exhaustiveness, "n_poses": 1,
        "conformers": 1, "timeout": args.timeout,
    }
    oracle = docking.DockingOracle(configuration, seed=args.seed, cpu=args.cpu)
    records = []
    start = time.monotonic()
    with args.output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Catalog ID", "smiles", "docking_score"])
        writer.writeheader()
        for index, row in enumerate(library.itertuples(index=False), 1):
            record = oracle.evaluate(row.compound_id, row.smiles, args.artifacts / "poses")
            if record["status"] == "ok":
                writer.writerow({"Catalog ID": row.compound_id, "smiles": row.smiles, "docking_score": record["score"]})
                handle.flush()
                record["pose_sha256"] = file_hash(record["pose"])
                record["pose"] = portable(record["pose"])
                for conformer in record.get("conformers", []):
                    conformer["pose_sha256"] = file_hash(conformer["pose"])
                    conformer["pose"] = portable(conformer["pose"])
            record.update({"smiles": row.smiles, "source_row": int(row.source_row)})
            records.append(record)
            (args.artifacts / "records.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
            print(f"{index}/{args.count} {row.compound_id}: {record['status']} {record['score']}", flush=True)
    failures = [record for record in records if record["status"] != "ok"]
    (args.artifacts / "failures.json").write_text(json.dumps(failures, indent=2, allow_nan=False) + "\n")
    provenance = {
        "description": "Actual AutoDock Vina energies for a fixed prefix; a computational test fixture, not experimental affinities",
        "selection": "smallest valid ID per canonical isomeric SMILES; first N representatives in source-record order; no score-based selection",
        "requested": args.count, "succeeded": len(records) - len(failures), "failed": len(failures),
        "input_rejections": len(rejected), "seed": args.seed,
        "configuration": oracle.parameters | {"timeout": oracle.timeout},
        "inputs": {portable(path): file_hash(path) for path in (args.input, args.receptor, args.reference)},
        "output": portable(args.output), "output_sha256": file_hash(args.output),
        "artifacts": portable(args.artifacts),
        "generator_sha256": file_hash(__file__),
        "oracle_sha256": file_hash(docking.__file__),
        "command": [str(Path(sys.argv[0])), *sys.argv[1:]],
        "versions": {name: version(name) for name in ("vina", "meeko", "rdkit", "numpy")},
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "elapsed_seconds": time.monotonic() - start, "records": records,
    }
    provenance["configuration"]["receptor"] = portable(args.receptor)
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"succeeded": provenance["succeeded"], "failed": len(failures),
                      "elapsed_seconds": provenance["elapsed_seconds"], "csv_sha256": provenance["output_sha256"]}))
    if failures:
        raise SystemExit("Some compounds failed; inspect failures.json. No replacement scores were generated.")


if __name__ == "__main__":
    main()
