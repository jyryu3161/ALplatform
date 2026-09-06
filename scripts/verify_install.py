#!/usr/bin/env python3
"""Verify an installed environment without downloading weights unless requested."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timezone
from importlib import import_module, metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
CORE_MODELS = ["tabm", "random_forest", "extra_trees"]
IMPORTS = {
    "numpy": ("numpy", None), "pandas": ("pandas", None), "scipy": ("scipy", None),
    "scikit-learn": ("sklearn.ensemble", "RandomForestRegressor"),
    "tabm": ("tabm", "TabM"), "rdkit": ("rdkit.Chem", "MolFromSmiles"),
    "synthefy-nori": ("synthefy_nori", "NoriRegressor"),
    "Xiaomi-TabLDM": ("tabldm", "TabLDMRegressor"), "tabfm": ("tabfm", "TabFMRegressor"),
    "safetensors": ("safetensors.torch", None), "vina": ("vina", "Vina"),
    "meeko": ("meeko", "MoleculePreparation"), "gemmi": ("gemmi", None),
    "pdbfixer": ("pdbfixer", "PDBFixer"), "openmm": ("openmm", "System"),
    "optuna": ("optuna", "create_study"), "pytest": ("pytest", "main"),
    "ruff": ("ruff", None), "build": ("build", "ProjectBuilder"),
}


def resource_config(backend: str):
    """Choose one permitted physical GPU, respecting an inherited mask."""
    from active_learning_docking.config import ResourceConfig

    if backend == "cpu":
        return ResourceConfig(cpu_threads=2, gpu_ids=[0, 1], device="cpu")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
    tokens = [token.strip() for token in mask.split(",") if token.strip()]
    if any(not token.isdigit() for token in tokens):
        raise RuntimeError("CUDA verification requires a numeric CUDA_VISIBLE_DEVICES mask containing GPU 0 or 1")
    allowed = [gpu for gpu in (0, 1) if str(gpu) in tokens]
    if not allowed:
        raise RuntimeError("CUDA verification requires an allowed physical GPU 0 or 1")
    return ResourceConfig(cpu_threads=2, gpu_ids=[allowed[0]], device=f"cuda:{allowed[0]}")


def check_torch_backend(torch, backend: str, device: str) -> None:
    expected_cuda = None if backend == "cpu" else "12.8"
    if torch.__version__.split("+")[0] != "2.8.0" or torch.version.cuda != expected_cuda:
        raise RuntimeError(f"Expected PyTorch 2.8.0 with backend {backend}; found "
                           f"{torch.__version__}, CUDA build {torch.version.cuda}")
    if backend != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("The cu128 installation requires a usable physical GPU 0 or 1")
        value = torch.arange(4, device=device).square().sum().item()
        torch.cuda.synchronize(device)
        if value != 14:
            raise RuntimeError("CUDA tensor operation produced an incorrect result")


def verify_runtime(backend: str) -> dict:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Installation verification requires Python 3.11; found {sys.version.split()[0]}")
    from active_learning_docking.resources import configure_resources

    # This installs affinity, native thread limits and the GPU mask before
    # configure_resources imports NumPy or PyTorch, including in child processes.
    requested = resource_config(backend)
    runtime = configure_resources(requested, seed=42)
    torch = import_module("torch")
    check_torch_backend(torch, backend, runtime["device"])
    versions = {"torch": metadata.version("torch")}
    for distribution, (module, attribute) in IMPORTS.items():
        loaded = import_module(module)
        if attribute is not None:
            getattr(loaded, attribute)
        versions[distribution] = metadata.version(distribution)
    getattr(import_module("sklearn.ensemble"), "ExtraTreesRegressor")
    getattr(import_module("tabfm").tabfm_v1_0_0_pytorch, "load")
    return {"python": sys.version.split()[0], "versions": versions,
            "resources": {key: value for key, value in runtime.items() if key != "threadpool_controller"},
            "requested_resources": {"cpu_threads": 2, "gpu_ids": requested.gpu_ids, "device": requested.device}}


def run_logged(command: list[str], log_path: Path, *, environment=None) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed with exit {result.returncode}; see {log_path}")


def validate_smoke_outputs(directory: Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("successful") != 12 or summary.get("attempted") != 12 or summary.get("models") != CORE_MODELS:
        raise RuntimeError("Replay smoke test did not complete 12 successful acquisitions with all three core models")
    with (directory / "predictions.csv").open(newline="") as source:
        predictions = list(csv.DictReader(source))
    if len(predictions) != 48 or any(
        not math.isfinite(float(row[key])) for row in predictions for key in ("prediction", "uncertainty")
    ) or any(float(row["uncertainty"]) < 0 for row in predictions):
        raise RuntimeError("Replay smoke test returned missing or non-finite predictions or invalid uncertainty")
    return {"successful": 12, "prediction_rows": len(predictions), "models": summary["models"]}


def smoke_test(directory: Path, resources: dict) -> dict:
    output = directory / "replay"
    configuration = {
        "schema_version": 1, "seed": 42, "mode": "replay", "output_dir": str(output),
        "resources": {**resources, "deterministic": True},
        "data": {"path": str(ROOT / "examples/replay.csv"), "score_column": "synthetic_score",
                 "fingerprint_bits": 128},
        "model": {"models": CORE_MODELS, "ensemble": {"strategy": "mean"},
                  "n_blocks": 1, "d_block": 16, "k": 4, "epochs": 3, "patience": 2, "batch_size": 16,
                  "params": {"random_forest": {"n_estimators": 8}, "extra_trees": {"n_estimators": 8}}},
        "learning": {"initial_size": 8, "batch_size": 4, "cycles": 1, "budget": 12, "strategy": "lcb"},
        "evaluation": {"holdout_fraction": .2, "top_k": 4},
    }
    path = directory / "smoke.yaml"
    path.write_text(json.dumps(configuration, indent=2) + "\n")  # JSON is valid YAML.
    run_logged([sys.executable, "-m", "active_learning_docking", "run", "--config", str(path)],
               directory / "smoke.log")
    return validate_smoke_outputs(output)


def run_tests(directory: Path, *, test_models: bool) -> dict:
    environment = os.environ.copy()
    environment.update(ALD_RUN_DOCKING_TESTS="1", ALD_RUN_FOUNDATION_TESTS="1" if test_models else "0")
    environment.pop("PYTEST_ADDOPTS", None)
    report_path = directory / "pytest.xml"
    run_logged([sys.executable, "-m", "pytest", "-q", "-ra", str(ROOT / "tests"),
                f"--junitxml={report_path}"], directory / "pytest.log", environment=environment)
    cases = ET.parse(report_path).findall(".//testcase")
    required = {"test_real_docking_reproducibility[vina]", "test_real_docking_reproducibility[qvina]",
                "test_real_multiple_conformers_and_poses"}
    if test_models:
        required |= {f"test_real_public_foundation_inference_and_restore[{name}]"
                     for name in ("nori", "tabldm", "tabfm")}
    passed = {case.attrib["name"] for case in cases if not any(
        case.find(tag) is not None for tag in ("skipped", "failure", "error"))}
    if not required <= passed:
        raise RuntimeError(f"Required native integration tests did not pass: {sorted(required - passed)}")
    return {"tests": len(cases), "skipped": sum(case.find("skipped") is not None for case in cases),
            "real_docking": True, "real_foundation_models": test_models}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-backend", choices=("cpu", "cu128"), default="cpu")
    parser.add_argument("--test", action="store_true", help="Run the full test suite, including actual docking")
    parser.add_argument("--test-models", action="store_true", help="Also test real foundation weights; implies --test")
    parser.add_argument("--offline", action="store_true", help="Use only cached foundation checkpoints")
    args = parser.parse_args(argv)
    if sys.prefix == sys.base_prefix:
        print("Run verification with the installed virtual environment's Python", file=sys.stderr)
        return 1
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    parent = Path(sys.prefix) / "installation-checks"
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-"), dir=parent))
    report = {"status": "running", "torch_backend": args.torch_backend, "offline": args.offline,
              "directory": str(directory)}
    result = 0
    try:
        print(f"Checking installed runtimes; logs: {directory}", flush=True)
        with (directory / "runtime.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            report["runtime"] = verify_runtime(args.torch_backend)
        print("Running the three-model replay smoke test", flush=True)
        report["smoke"] = smoke_test(directory, report["runtime"]["requested_resources"])
        if args.test or args.test_models:
            print("Running the full test suite" + (" with real foundation weights" if args.test_models else ""),
                  flush=True)
            report["tests"] = run_tests(directory, test_models=args.test_models)
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="failed", error=str(exc) or type(exc).__name__)
        (directory / "failure.log").write_text(traceback.format_exc())
        print(f"Installation verification failed: {report['error']}", file=sys.stderr)
        result = 130 if isinstance(exc, KeyboardInterrupt) else 1
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Verification {report['status']}: {directory / 'report.json'}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
