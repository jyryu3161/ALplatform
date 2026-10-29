"""Independent-process repeatability checks and paired-seed replay benchmarks."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Sequence

import yaml

from .provenance import artifact_hashes, write_json


def _new_output(path: str | Path) -> Path:
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run(config: Any, root: Path, name: str) -> dict[str, Any]:
    """Execute one configuration in a fresh interpreter and return its summary."""
    config = copy.deepcopy(config)
    config.output_dir = str(root / name)
    config_path = root / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    with (root / f"{name}.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run([sys.executable, "-m", "active_learning_docking", "run",
                                    "--config", str(config_path)], stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"{name} failed (exit {completed.returncode}); see {root / (name + '.log')}")
    return json.loads((root / name / "summary.json").read_text())


def compare_runs(first: str | Path, second: str | Path) -> dict[str, Any]:
    """Compare scientific records, predictions and pose bytes (.pdbqt, .sdf); exclude times/paths.

    Model serialization need not be byte-identical across backends. Saved model
    predictions are compared independently during tests. No numeric tolerance is
    used here: any score, ordering, uncertainty or metric difference fails. A
    record present in only one run fails; ``split.csv`` and ``predictions.csv``
    may be absent from both (no holdout, or ``save_predictions: false``).
    """
    first, second = Path(first).resolve(), Path(second).resolve()
    inventories = []
    for directory in (first, second):
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["status"] != "completed":
            raise ValueError(f"run is not completed: {directory}")
        inventory = artifact_hashes(directory)
        if inventory != manifest["artifact_sha256"]:
            raise ValueError(f"artifact integrity mismatch (inventory or content): {directory}")
        inventories.append(inventory)
    canonical_names = ["observations.csv", "final_results.csv", "predictions.csv", "split.csv",
                       "metrics.csv", "training_history.csv", "summary.json"]
    cycle_names = {
        str(path.relative_to(root))
        for root in (first, second)
        for pattern in ("cycle_*/*.csv", "cycle_*/metrics.json")
        for path in root.glob(pattern)
    }
    canonical_names += sorted(cycle_names)
    optional = {"split.csv", "predictions.csv"}
    comparisons = {}
    for name in canonical_names:
        a, b = first / name, second / name
        if name in optional and not a.is_file() and not b.is_file():
            continue  # Records that a configuration legitimately omits from both runs.
        if not a.is_file() or not b.is_file():
            comparisons[name] = False
            continue
        if a.suffix == ".csv":
            with a.open(newline="", encoding="utf-8") as handle:
                left = list(csv.reader(handle))
            with b.open(newline="", encoding="utf-8") as handle:
                right = list(csv.reader(handle))
            for records, root in ((left, first), (right, second)):
                pose_columns = [index for index, name in enumerate(records[0]) if name == "pose"] if records else []
                for row in records[1:]:
                    for index in pose_columns:
                        if index < len(row) and row[index]:
                            row[index] = str(Path(row[index]).relative_to(root))
            comparisons[name] = left == right
        else:
            comparisons[name] = a.read_bytes() == b.read_bytes()
    # The verified inventories already contain byte hashes for every pose
    # (PDBQT docking poses and SDF GNINA rescoring inputs/outputs).
    # Reuse them rather than rereading potentially large checkpoint/pose files.
    left_poses, right_poses = [{name: digest for name, digest in inventory.items()
                                if name.endswith((".pdbqt", ".sdf"))}
                               for inventory in inventories]
    comparisons["all_pose_bytes"] = left_poses == right_poses
    return {"passed": all(comparisons.values()), "comparison": "exact", "checks": comparisons}


def reproduce(config: Any, output: str | Path) -> dict[str, Any]:
    """Run the same configuration twice in fresh processes and verify results."""
    root = _new_output(output)
    _run(config, root, "repeat_a")
    _run(config, root, "repeat_b")
    report = compare_runs(root / "repeat_a", root / "repeat_b")
    report.update(seed=config.seed, mode=config.mode, device=config.resources.device)
    write_json(root / "reproducibility.json", report)
    if not report["passed"]:
        raise RuntimeError(f"reproducibility failed; see {root / 'reproducibility.json'}")
    return report


def benchmark(config: Any, output: str | Path, seeds: Sequence[int],
              strategies: Sequence[str]) -> dict[str, Any]:
    """Compare acquisition policies at matched budgets and fixed paired splits."""
    if config.mode != "replay":
        raise ValueError("benchmark requires replay scores for an exhaustive reference")
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("benchmark requires at least two distinct seeds")
    if len(set(strategies)) != len(strategies) or not strategies:
        raise ValueError("benchmark strategies must be unique and nonempty")
    root = _new_output(output)
    rows = []
    for seed in seeds:
        for strategy in strategies:
            candidate = copy.deepcopy(config)
            candidate.seed, candidate.learning.strategy = seed, strategy
            candidate.validate()
            summary = _run(candidate, root, f"{strategy}_seed_{seed}")
            rows.append({"seed": seed, "strategy": strategy, **summary["final_metrics"]})
    with (root / "paired_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Descriptive statistics only; multiple seeds do not replace multiple targets.
    aggregates: dict[str, dict[str, Any]] = {}
    keys = ["discovery_top_k_recall", "discovery_enrichment_factor", "discovery_simple_regret", "holdout_rmse"]
    for strategy in strategies:
        aggregates[strategy] = {}
        for key in keys:
            values = [r[key] for r in rows if r["strategy"] == strategy and r.get(key) is not None]
            if values:
                aggregates[strategy][key] = {"mean": statistics.mean(values),
                                             "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                                             "n_seeds": len(values)}
    report = {"seeds": seeds, "strategies": strategies, "aggregate": aggregates,
              "interpretation": "Descriptive paired-seed replay; no claim of biological or cross-target superiority."}
    write_json(root / "benchmark.json", report)
    return report
