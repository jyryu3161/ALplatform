"""Installation checks fail clearly and exercise the actual installed core API."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def verifier():
    spec = importlib.util.spec_from_file_location("verify_install", ROOT / "scripts/verify_install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mask,physical", [("0,1", 0), ("1", 1), ("7,1", 1)])
def test_cuda_verification_respects_allowed_inherited_device(verifier, monkeypatch, mask, physical):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    resources = verifier.resource_config("cu128")
    assert resources.device == f"cuda:{physical}"
    assert resources.gpu_ids == [physical]
    assert resources.cpu_threads == 2


@pytest.mark.parametrize("mask", ["", "7", "GPU-uuid"])
def test_cuda_verification_rejects_unavailable_physical_devices(verifier, monkeypatch, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(RuntimeError, match="GPU 0 or 1"):
        verifier.resource_config("cu128")
    assert verifier.resource_config("cpu").device == "cpu"


@pytest.mark.parametrize("version,cuda,requested", [
    ("2.8.0+cu128", "12.8", "cpu"), ("2.8.0+cpu", None, "cu128"), ("2.9.0+cpu", None, "cpu"),
])
def test_wrong_torch_build_fails_before_tensor_allocation(verifier, version, cuda, requested):
    torch = SimpleNamespace(__version__=version, version=SimpleNamespace(cuda=cuda))
    with pytest.raises(RuntimeError, match="Expected PyTorch 2.8.0"):
        verifier.check_torch_backend(torch, requested, "cpu")


def test_smoke_runs_real_core_ensemble_from_another_directory(verifier, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = verifier.smoke_test(tmp_path, {"cpu_threads": 2, "gpu_ids": [0, 1], "device": "cpu"})
    assert report == {"successful": 12, "prediction_rows": 48, "models": verifier.CORE_MODELS}
    model = json.loads((tmp_path / "replay/model/model_metadata.json").read_text())["model_info"]
    assert model["models"] == verifier.CORE_MODELS
    assert model["weights"] == dict.fromkeys(verifier.CORE_MODELS, 1 / 3)


@pytest.mark.parametrize("bad_value", ["nan", "inf", "-0.1"])
def test_smoke_rejects_invalid_uncertainty(verifier, tmp_path, bad_value):
    (tmp_path / "summary.json").write_text(json.dumps({
        "successful": 12, "attempted": 12, "models": verifier.CORE_MODELS,
    }))
    with (tmp_path / "predictions.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["prediction", "uncertainty"])
        writer.writerows([[-3, bad_value]] * 48)
    with pytest.raises(RuntimeError, match="invalid uncertainty"):
        verifier.validate_smoke_outputs(tmp_path)


def test_runtime_failure_is_reported_and_retry_preserves_evidence(verifier, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(verifier.sys, "prefix", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")

    def missing_runtime(backend):
        assert verifier.os.environ["HF_HUB_OFFLINE"] == "1"
        raise ImportError("missing selected native runtime")

    monkeypatch.setattr(verifier, "verify_runtime", missing_runtime)
    assert verifier.main(["--offline"]) == 1
    first = next((tmp_path / "installation-checks").iterdir())
    evidence = (first / "report.json").read_bytes()
    assert json.loads(evidence)["status"] == "failed"
    assert "missing selected native runtime" in (first / "failure.log").read_text()
    assert "Verification passed" not in capsys.readouterr().out
    monkeypatch.setattr(verifier, "verify_runtime", lambda backend: {"requested_resources": {}})
    monkeypatch.setattr(verifier, "smoke_test", lambda *args: {"successful": 12})
    assert verifier.main([]) == 0
    assert len(list((tmp_path / "installation-checks").iterdir())) == 2
    assert (first / "report.json").read_bytes() == evidence


def test_subprocess_failure_keeps_diagnostic_log(verifier, tmp_path):
    log = tmp_path / "failed command.log"
    with pytest.raises(RuntimeError, match="exit 23"):
        verifier.run_logged([verifier.sys.executable, "-c", "print('diagnostic'); raise SystemExit(23)"], log)
    assert log.read_text().strip() == "diagnostic"


@pytest.mark.parametrize("test_models", [False, True])
def test_full_suite_enables_only_requested_native_tests(verifier, tmp_path, monkeypatch, test_models):
    monkeypatch.setenv("ALD_RUN_FOUNDATION_TESTS", "inherited")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k exclude_everything")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def run(command, log_path, *, environment):
        assert environment["ALD_RUN_DOCKING_TESTS"] == "1"
        assert environment["ALD_RUN_FOUNDATION_TESTS"] == str(int(test_models))
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert "PYTEST_ADDOPTS" not in environment
        suite = ET.Element("testsuite")
        for name in ["test_real_docking_reproducibility[vina]", "test_real_docking_reproducibility[qvina]",
                     "test_real_multiple_conformers_and_poses"]:
            ET.SubElement(suite, "testcase", name=name)
        for name in verifier.FOUNDATION_TESTS:
            case = ET.SubElement(suite, "testcase", name=name)
            if not test_models:
                ET.SubElement(case, "skipped")
        ET.ElementTree(suite).write(tmp_path / "pytest.xml")

    monkeypatch.setattr(verifier, "run_logged", run)
    result = verifier.run_tests(tmp_path, test_models=test_models)
    assert result["tests"] == 8 and result["skipped"] == (0 if test_models else 5)
    assert verifier.os.environ["ALD_RUN_FOUNDATION_TESTS"] == "inherited"


def test_model_check_requires_real_tabpfn_weight_tests(verifier, tmp_path, monkeypatch):
    assert verifier.IMPORTS["tabpfn"] == ("tabpfn", "TabPFNRegressor")
    assert {"test_real_public_foundation_inference_and_restore[tabpfn]",
            "test_real_tabpfn_replay_runs_are_exact_and_holdout_blind"} <= set(verifier.FOUNDATION_TESTS)
    suite = ET.Element("testsuite")
    for name in ["test_real_docking_reproducibility[vina]", "test_real_docking_reproducibility[qvina]",
                 "test_real_multiple_conformers_and_poses", *verifier.FOUNDATION_TESTS]:
        case = ET.SubElement(suite, "testcase", name=name)
        if name == "test_real_tabpfn_replay_runs_are_exact_and_holdout_blind":
            ET.SubElement(case, "skipped")
    ET.ElementTree(suite).write(tmp_path / "pytest.xml")
    monkeypatch.setattr(verifier, "run_logged", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="test_real_tabpfn_replay_runs_are_exact_and_holdout_blind"):
        verifier.run_tests(tmp_path, test_models=True)


def test_required_native_tests_cannot_silently_skip(verifier, tmp_path, monkeypatch):
    (tmp_path / "pytest.xml").write_text(
        '<testsuite><testcase name="test_real_docking_reproducibility[vina]"><skipped/></testcase></testsuite>'
    )
    monkeypatch.setattr(verifier, "run_logged", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="Required native integration tests did not pass"):
        verifier.run_tests(tmp_path, test_models=False)


@pytest.fixture
def installer(tmp_path):
    """Run the real Bash entrypoints against an isolated, recording uv implementation."""
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    for name in ("install.sh", "setup_server.sh"):
        shutil.copyfile(ROOT / name, checkout / name)
    (checkout / "requirements").mkdir()
    pins = "\n".join(f"{name}==1.0" for name in (
        "setuptools", "wheel", "packaging", "hatchling", "pathspec", "pluggy", "tomlkit", "trove-classifiers",
    ))
    for backend in ("cpu", "cu128"):
        (checkout / f"requirements/install-{backend}.lock").write_text(pins + "\n")
    binaries = tmp_path / "binaries"
    binaries.mkdir()
    for name in ("bash", "dirname", "uname", "getconf", "realpath", "cat", "git"):
        binaries.joinpath(name).symlink_to(shutil.which(name))
    runner = tmp_path / "fake commands.py"
    runner.write_text('''import json, os, pathlib, sys
mode, arguments = sys.argv[1], sys.argv[2:]
with open(os.environ["INSTALL_CALLS"], "a") as log:
    log.write(json.dumps({"mode": mode, "arguments": arguments,
                         "active_env": os.environ.get("VIRTUAL_ENV"),
                         "clear": os.environ.get("UV_VENV_CLEAR")}) + "\\n")
if mode == "uv":
    if arguments == ["--version"]:
        print("uv 0.11.7")
        raise SystemExit(0)
    while arguments and arguments[0] in ("--no-config", "--offline"):
        arguments.pop(0)
    if arguments[0] == "venv":
        target = pathlib.Path(arguments[-1])
        (target / "bin").mkdir(parents=True)
        (target / "bin/python").write_text(os.environ["FAKE_PYTHON_SCRIPT"])
        (target / "bin/python").chmod(0o755)
        (target / "pyvenv.cfg").write_text("include-system-site-packages = false\\n")
    elif arguments[:2] == ["pip", "install"]:
        raise SystemExit(int(os.environ.get("FAKE_INSTALL_EXIT", "0")))
    elif arguments[:2] == ["pip", "freeze"]:
        print("fixture==1.0")
elif arguments[:2] == ["-I", "-"]:
    expected, target = arguments[2:]
    sys.version = os.environ.get("FAKE_PYTHON_VERSION", expected) + " (test interpreter)"
    sys.prefix, sys.base_prefix = target, str(pathlib.Path(target).parent / "managed-python")
    sys.argv = ["-", expected, target]
    exec(sys.stdin.read(), {"__name__": "__main__"})
else:
    raise SystemExit(int(os.environ.get("FAKE_VERIFY_EXIT", "0")))
''')
    wrapper = '#!/bin/bash\nexec "$INSTALL_RUNNER" "$INSTALL_IMPLEMENTATION" '
    python_script = wrapper + 'python "$@"\n'
    uv = binaries / "uv"
    uv.write_text(wrapper + 'uv "$@"\n')
    uv.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    environment = {**os.environ, "PATH": str(binaries), "XDG_CACHE_HOME": str(tmp_path / "empty-cache"),
                   "INSTALL_RUNNER": sys.executable, "INSTALL_IMPLEMENTATION": str(runner),
                   "INSTALL_CALLS": str(calls), "FAKE_PYTHON_SCRIPT": python_script}

    def run(*arguments, wrapper=False, **overrides):
        return subprocess.run([str(binaries / "bash"), str(checkout / (
            "setup_server.sh" if wrapper else "install.sh")), *map(str, arguments)],
            cwd=tmp_path, env={**environment, **overrides}, capture_output=True, text=True, timeout=30)

    def recorded():
        return [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []

    def owned(target):
        (target / "bin").mkdir(parents=True)
        (target / "bin/python").write_text(python_script)
        (target / "bin/python").chmod(0o755)
        (target / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
        (target / ".ald-install-owner").write_text(str(checkout) + "\n")

    return SimpleNamespace(run=run, recorded=recorded, owned=owned, checkout=checkout,
                           binaries=binaries, calls=calls)


@pytest.mark.parametrize("arguments", [("--venv",), ("--torch-backend", "wrong"), ("--unknown",)])
def test_installer_rejects_invalid_arguments_before_uv(installer, arguments):
    result = installer.run(*arguments)
    assert result.returncode != 0 and "Installation failed" in result.stderr
    assert installer.recorded() == []
    assert not (installer.checkout / ".venv-ald").exists()


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling_symlink", "unmanaged_venv"])
def test_installer_rejects_existing_unowned_targets_without_mutation(installer, tmp_path, kind):
    target = tmp_path / "existing environment"
    if kind == "file":
        target.write_text("existing content")
    elif kind in {"symlink", "dangling_symlink"}:
        destination = tmp_path / "original target"
        if kind == "symlink":
            destination.mkdir()
        target.symlink_to(destination, target_is_directory=True)
    else:
        target.mkdir()
        (target / "sentinel").write_text("existing content")
        if kind == "unmanaged_venv":
            (target / "pyvenv.cfg").write_text("include-system-site-packages = false\n")

    def snapshot():
        paths = [target] + (list(target.rglob("*")) if target.is_dir() and not target.is_symlink() else [])
        return {str(path): (os.readlink(path) if path.is_symlink() else
                            path.read_bytes() if path.is_file() else "directory") for path in paths}

    before = snapshot()
    assert installer.run("--venv", target).returncode != 0
    assert snapshot() == before and installer.recorded() == []


def test_installer_rejects_wrong_python_before_uv(installer, tmp_path):
    target = tmp_path / "owned environment"
    installer.owned(target)
    before = {str(path): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    result = installer.run("--venv", target, FAKE_PYTHON_VERSION="3.10.99")
    assert result.returncode != 0 and "Expected isolated Python 3.11.13" in result.stderr
    assert all(call["mode"] != "uv" for call in installer.recorded())
    assert before == {str(path): path.read_bytes() for path in target.rglob("*") if path.is_file()}


def test_offline_missing_uv_never_bootstraps_or_creates_environment(installer):
    (installer.binaries / "uv").unlink()
    result = installer.run("--offline")
    assert result.returncode != 0 and "not available locally" in result.stderr
    assert installer.recorded() == []
    assert not (installer.checkout / ".venv-ald").exists()


def test_quoted_paths_and_rerun_preserve_owned_environment(installer, tmp_path):
    target = tmp_path / "new environment with spaces"
    options = ("--venv", target, "--offline", "--test")
    assert installer.run(*options, VIRTUAL_ENV="unrelated", UV_VENV_CLEAR="1").returncode == 0
    sentinel = target / "unrelated-installed-package"
    sentinel.write_bytes(b"preserve this")
    first_calls = len(installer.recorded())
    assert installer.run(*options).returncode == 0
    assert sentinel.read_bytes() == b"preserve this"
    calls = installer.recorded()
    assert sum("venv" in call["arguments"] for call in calls if call["mode"] == "uv") == 1
    assert not any("venv" in call["arguments"] for call in calls[first_calls:] if call["mode"] == "uv")
    assert all(call["active_env"] is None and call["clear"] is None for call in calls)
    for call in calls:
        arguments = call["arguments"]
        if call["mode"] == "uv" and arguments != ["--version"]:
            assert "--offline" in arguments
            assert "--exact" not in arguments and "--clear" not in arguments
        if call["mode"] == "uv" and "--python" in arguments and "pip" in arguments:
            assert arguments[arguments.index("--python") + 1] == str(target / "bin/python")
    helper = [call for call in calls if call["mode"] == "python" and call["arguments"][0] != "-I"]
    assert helper[-1]["arguments"] == [str(installer.checkout / "scripts/verify_install.py"),
                                      "--torch-backend", "cpu", "--offline", "--test"]


def test_failed_install_can_retry_without_recreating_environment(installer, tmp_path):
    target = tmp_path / "retry environment"
    result = installer.run("--venv", target, FAKE_INSTALL_EXIT="23")
    assert result.returncode == 23 and "succeeded" not in result.stdout
    sentinel = target / "partial-install-evidence"
    sentinel.write_text("preserve")
    assert installer.run("--venv", target).returncode == 0
    assert sentinel.read_text() == "preserve"
    assert sum("venv" in call["arguments"] for call in installer.recorded() if call["mode"] == "uv") == 1


def test_server_wrapper_forwards_arguments_and_verification_failure(installer, tmp_path):
    target = tmp_path / "wrapper environment"
    result = installer.run("--venv", target, "--torch-backend", "cu128", "--test-models", "--offline",
                           wrapper=True, FAKE_VERIFY_EXIT="19")
    assert result.returncode == 19 and "succeeded" not in result.stdout
    helper = [call for call in installer.recorded() if call["mode"] == "python" and call["arguments"][0] != "-I"]
    assert helper[-1]["arguments"][1:] == ["--torch-backend", "cu128", "--test-models", "--offline"]
