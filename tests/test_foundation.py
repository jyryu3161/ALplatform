"""Contracts for optional backends; real downloadable-model smoke runs are separate."""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from active_learning_docking import foundation
from active_learning_docking.foundation import FoundationRegressor, validate_foundation_params
from active_learning_docking.store import PackedFingerprints


class FakeRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.batches = []

    def fit(self, X, y, **kwargs):
        self.fit_kwargs = kwargs
        self.X, self.y = np.array(X), np.array(y)
        return self

    def predict(self, X, output_type=None):
        self.batches.append(np.array(X))
        mean = X.sum(axis=1, dtype=np.float64) + self.y.mean()
        quantiles = mean[:, None] + np.array([-2., 0., 2.])
        if output_type == "full":
            return {"mean": mean, "quantiles": quantiles, "taus": [.25, .5, .75]}
        if output_type == ["mean", "raw_quantiles"]:
            return {"mean": mean, "raw_quantiles": quantiles}
        return mean


class FakeTabFMRegressor(FakeRegressor):
    """Keep the routed PyTorch options explicit instead of swallowing unknown keys."""

    def __init__(self, model, n_estimators=32, norm_methods=None, max_num_features=500,
                 max_num_rows=None, batch_size=1, random_state=42):
        super().__init__(model=model, n_estimators=n_estimators, norm_methods=norm_methods,
                         max_num_features=max_num_features, max_num_rows=max_num_rows,
                         batch_size=batch_size, random_state=random_state)


@pytest.fixture
def fake_packages(monkeypatch, tmp_path):
    loads = []

    def loader(**kwargs):
        loads.append(kwargs)
        return object()

    packages = {
        "synthefy_nori": SimpleNamespace(NoriRegressor=FakeRegressor),
        "tabldm": SimpleNamespace(TabLDMRegressor=FakeRegressor),
        "tabfm": SimpleNamespace(TabFMRegressor=FakeTabFMRegressor,
                                  tabfm_v1_0_0_pytorch=SimpleNamespace(load=loader)),
        "safetensors.torch": SimpleNamespace(),
    }
    monkeypatch.setattr(foundation, "import_module", lambda name: packages[name])
    monkeypatch.setattr(foundation, "sys", SimpleNamespace(version_info=(3, 11)))
    checkpoint = tmp_path / "fake-weights"
    checkpoint.write_bytes(b"test-only checkpoint")

    def resolve(self, options):
        self.checkpoint_reference_ = {"path": str(checkpoint), "repository": None, "revision": None,
                                      "files": [{"path": str(checkpoint),
                                                 "sha256": self._file_hash(checkpoint)}]}
        return options

    monkeypatch.setattr(FoundationRegressor, "_resolve_checkpoint", resolve)
    return loads


@pytest.mark.parametrize("backend", ["nori", "tabldm", "tabfm"])
def test_local_api_contract_uncertainty_and_empty_queries(fake_packages, backend):
    X = np.arange(32).reshape(8, 4)
    model = FoundationRegressor(backend, inference_batch_size=2).fit(X, np.arange(8))
    mean, std = model.predict(X[:5])
    np.testing.assert_allclose(mean, X[:5].sum(axis=1) + 3.5)
    np.testing.assert_allclose(std, 0 if backend == "tabfm" else np.sqrt(3 if backend == "nori" else 8/3))
    assert [len(batch) for batch in model.model.batches] == [2, 2, 1]
    assert model.validation_indices_.size == 0
    assert model.best_validation_loss_ is None
    assert model.history_[0]["weight_updates"] == 0
    assert model.predict(X[:0])[0].shape == (0,)
    np.testing.assert_equal(model.predict(X[:1], return_std=False), mean[:1])
    if backend == "nori":
        assert model.model.kwargs["token"] is False
        assert model.model.kwargs["device"] == "cpu"
        assert model.model.kwargs["model"] == "nori-6m"
    elif backend == "tabldm":
        assert model.model.kwargs["n_jobs"] is None
        assert model.model.kwargs["random_state"] == 42
    else:
        assert fake_packages == [{"model_type": "regression", "checkpoint_path": None, "device": "cpu"}]
        assert model.model.kwargs["max_num_features"] == 128


def test_context_only_selection_and_lazy_pool(fake_packages, monkeypatch):
    rng = np.random.default_rng(2)
    X = rng.integers(0, 2, (30, 20), dtype=np.uint8)
    packed = PackedFingerprints.from_dense(X)
    model = FoundationRegressor("nori", params={"max_context_samples": 7, "max_features": 4},
                                seed=13, inference_batch_size=3).fit(packed, np.arange(30))
    expected_rows = np.sort(np.random.default_rng(13).choice(30, 7, replace=False))
    variance = X[expected_rows].var(axis=0)
    expected_features = np.sort(np.argsort(-variance, kind="stable")[:4])
    np.testing.assert_equal(model.train_indices_, expected_rows)
    np.testing.assert_equal(model.feature_indices_, expected_features)
    np.testing.assert_equal(model.model.X, X[expected_rows][:, expected_features])
    original = PackedFingerprints.__getitem__

    def bounded_getitem(self, key):
        result = original(self, key)
        assert len(result) <= 3
        return result

    monkeypatch.setattr(PackedFingerprints, "__getitem__", bounded_getitem)
    model.predict(packed)
    assert max(map(len, model.model.batches)) == 3
    # Queries never change selected features or the context used to fit.
    np.testing.assert_equal(model.feature_indices_, expected_features)


@pytest.mark.parametrize("backend", ["nori", "tabldm", "tabfm"])
def test_checkpoint_roundtrip_exact_context_without_resampling(fake_packages, backend, tmp_path):
    X = np.random.default_rng(4).normal(size=(17, 9))
    model = FoundationRegressor(backend, params={"max_context_samples": 5, "max_features": 3},
                                seed=23, inference_batch_size=4).fit(X, np.arange(17))
    model.save(tmp_path)
    restored = FoundationRegressor.load(tmp_path)
    np.testing.assert_equal(model.context_X_, restored.context_X_)
    np.testing.assert_equal(model.train_indices_, restored.train_indices_)
    np.testing.assert_equal(model.feature_indices_, restored.feature_indices_)
    for actual, expected in zip(restored.predict(X), model.predict(X)):
        np.testing.assert_equal(actual, expected)
    info = json.loads((tmp_path / "model_metadata.json").read_text())["model_info"]
    assert info["n_training"] == 5
    assert info["model_type"] == backend
    with open(tmp_path / "foundation_context.npz", "ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="SHA256"):
        FoundationRegressor.load(tmp_path)


@pytest.mark.parametrize("params", [
    {"api_key": "unused"}, {"device": "cuda:7"}, {"max_context_samples": 1},
    {"max_features": True}, {"model": "remote"}, {"max_features": float("nan")},
    {"model": []}, {"augmentations": [{}]}, {3: True},
])
def test_reject_invalid_or_hosted_configuration(params):
    with pytest.raises(ValueError):
        validate_foundation_params("nori", params)


@pytest.mark.parametrize("key", ["cache_context", "maybe_quantize_kv_cache", "keep_cache_on_device"])
def test_tabfm_rejects_unsupported_context_cache_flags_before_loading(monkeypatch, key):
    monkeypatch.setattr(foundation, "import_module", lambda *args: pytest.fail("must reject before imports"))
    monkeypatch.setattr(FoundationRegressor, "_resolve_checkpoint",
                        lambda *args: pytest.fail("must reject before downloading or loading weights"))
    with pytest.raises(ValueError, match=f"Unknown tabfm parameters.*{key}"):
        FoundationRegressor("tabfm", params={key: True})


def test_tabfm_missing_safetensors_fails_before_loading_weights(fake_packages, monkeypatch):
    modules = foundation.import_module

    def missing_safetensors(name):
        if name == "safetensors.torch":
            raise ImportError("safetensors is missing")
        return modules(name)

    monkeypatch.setattr(foundation, "import_module", missing_safetensors)
    model = FoundationRegressor("tabfm")
    with pytest.raises(ImportError, match="tabfm.*extra"):
        model.preflight()
    with pytest.raises(ImportError, match="tabfm.*extra"):
        model.fit(np.arange(8).reshape(4, 2), np.arange(4))
    assert fake_packages == []  # Google's model loader never allocated a network.


@pytest.mark.parametrize("files,missing", [
    (["config.json", "pytorch_model.bin"], "model.safetensors"),
    (["model.safetensors"], "config.json"),
])
def test_tabfm_local_checkpoint_preflight_requires_supported_files(monkeypatch, tmp_path, files, missing):
    def loader(**kwargs):
        pytest.fail("invalid local checkpoints must fail before model allocation")

    module = SimpleNamespace(TabFMRegressor=FakeTabFMRegressor,
                             tabfm_v1_0_0_pytorch=SimpleNamespace(load=loader))
    monkeypatch.setattr(foundation, "sys", SimpleNamespace(version_info=(3, 11)))
    monkeypatch.setattr(foundation, "import_module",
                        lambda name: module if name == "tabfm" else SimpleNamespace())
    for filename in files:
        (tmp_path / filename).write_bytes(b"test-only checkpoint")
    model = FoundationRegressor("tabfm", params={"checkpoint_path": str(tmp_path)})
    with pytest.raises(FileNotFoundError, match=missing):
        model.preflight()


def test_preflight_missing_package_local_paths_and_python_version(fake_packages, monkeypatch, tmp_path):
    model = FoundationRegressor("nori", params={"model_path": str(tmp_path / "missing.pt")})
    with pytest.raises(FileNotFoundError):
        model.preflight()

    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(foundation, "import_module", missing)
    with pytest.raises(ImportError, match="No model fallback"):
        FoundationRegressor("nori").preflight()
    monkeypatch.setattr(foundation, "sys", SimpleNamespace(version_info=(3, 10)))
    with pytest.raises(RuntimeError, match="Python >= 3.11"):
        FoundationRegressor("tabfm")


def test_fit_predict_validation_and_nonfinite(fake_packages):
    model = FoundationRegressor("nori")
    X = np.arange(12).reshape(4, 3)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)
    with pytest.raises(ValueError, match="two acquired"):
        model.fit(X[:1], [1])
    with pytest.raises(ValueError, match="groups"):
        model.fit(X, np.arange(4), groups=["x"])
    model.fit(X, np.arange(4))
    with pytest.raises(ValueError, match="expected 3"):
        model.predict(X[:, :2])
    model.model.y[:] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        model.predict(X)


def test_runtime_restores_global_rng_and_threads_on_error(fake_packages, monkeypatch):
    import random
    import torch

    model = FoundationRegressor("tabldm", cpu_threads=1).fit(np.arange(6).reshape(3, 2), np.arange(3))
    numpy_state, python_state, torch_state = np.random.get_state(), random.getstate(), torch.get_rng_state()
    threads = torch.get_num_threads()

    def broken(*args, **kwargs):
        assert torch.get_num_threads() == 1
        np.random.random()
        random.random()
        torch.rand(1)
        raise RuntimeError("upstream failure")

    monkeypatch.setattr(model.model, "predict", broken)
    with pytest.raises(RuntimeError, match="upstream failure"):
        model.predict(np.ones((1, 2)))
    np.testing.assert_equal(np.random.get_state()[1], numpy_state[1])
    assert random.getstate() == python_state
    assert torch.equal(torch_state, torch.get_rng_state())
    assert torch.get_num_threads() == threads


@pytest.mark.parametrize("cache", [True, "kv", "repr"])
def test_tabldm_cache_is_passed_to_fit(fake_packages, cache):
    model = FoundationRegressor("tabldm", params={"kv_cache": cache})
    model.fit(np.arange(6).reshape(3, 2), np.arange(3))
    assert model.model.fit_kwargs == {"kv_cache": cache}
    with pytest.raises(ValueError, match="kv_cache"):
        validate_foundation_params("tabldm", {"kv_cache": "auto"})


def test_tabldm_disk_offload_requires_a_directory():
    with pytest.raises(ValueError, match="requires disk_offload_dir"):
        validate_foundation_params("tabldm", {"offload_mode": "disk"})
    validate_foundation_params("tabldm", {"offload_mode": "disk", "disk_offload_dir": "/tmp/cache"})
    for params in ({"offload_mode": 1}, {"offload_mode": []}, {"feat_shuffle_method": []}):
        with pytest.raises(ValueError):
            validate_foundation_params("tabldm", params)


def test_restore_rejects_changed_package_and_pretrained_weights(fake_packages, monkeypatch, tmp_path):
    model = FoundationRegressor("nori").fit(np.arange(8).reshape(4, 2), np.arange(4))
    model.save(tmp_path / "saved")
    installed = foundation.version
    monkeypatch.setattr(foundation, "version", lambda _: "incompatible-version")
    with pytest.raises(ValueError, match="package version mismatch"):
        FoundationRegressor.load(tmp_path / "saved")
    monkeypatch.setattr(foundation, "version", installed)
    weights = model.checkpoint_reference_["files"][0]["path"]
    with open(weights, "ab") as stream:
        stream.write(b"changed pretrained weights")
    with pytest.raises(ValueError, match="Pretrained checkpoint SHA256 mismatch"):
        FoundationRegressor.load(tmp_path / "saved")


def test_public_checkpoint_can_move_to_a_different_hub_cache(fake_packages, monkeypatch, tmp_path):
    model = FoundationRegressor("nori").fit(np.arange(8).reshape(4, 2), np.arange(4))
    reference = model.checkpoint_reference_
    reference.update(repository="example/public-model", revision="fixed-revision")
    reference["files"][0]["hub_filename"] = "nori.pt"
    model.save(tmp_path / "saved")
    previous = reference["files"][0]["path"]
    moved = tmp_path / "new-cache.pt"
    os.rename(previous, moved)
    calls = []

    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(moved)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    # This check tests actual reference relocation, not the fake backend loader.
    monkeypatch.setattr(FoundationRegressor, "_fit_context", lambda self: setattr(self, "model", object()))
    restored = FoundationRegressor.load(tmp_path / "saved")
    assert restored.checkpoint_reference_["path"] == str(moved)
    assert calls == [("example/public-model", "nori.pt", {"revision": "fixed-revision", "token": False})]


def test_restore_rejects_different_git_build_with_same_package_version(fake_packages, monkeypatch, tmp_path):
    model = FoundationRegressor("tabldm").fit(np.arange(8).reshape(4, 2), np.arange(4))
    monkeypatch.setattr(FoundationRegressor, "_implementation_revision", lambda self: "first-commit")
    model.save(tmp_path)
    monkeypatch.setattr(FoundationRegressor, "_implementation_revision", lambda self: "different-commit")
    with pytest.raises(ValueError, match="source revision"):
        FoundationRegressor.load(tmp_path)


def test_partial_tabfm_cache_relocation_keeps_config_and_weights_together(fake_packages, monkeypatch, tmp_path):
    model = FoundationRegressor("tabfm").fit(np.arange(8).reshape(4, 2), np.arange(4))
    old_cache, new_cache = tmp_path / "old-cache", tmp_path / "new-cache"
    old_cache.mkdir()
    new_cache.mkdir()
    entries = []
    for name in ["config.json", "model.safetensors"]:
        file = old_cache / name
        file.write_text(name)
        (new_cache / name).write_text(name)
        entries.append({"path": str(file), "sha256": model._file_hash(file),
                        "hub_filename": f"regression/{name}"})
    model.checkpoint_reference_ = {"path": str(old_cache), "repository": "example/tabfm",
                                   "revision": "fixed-revision", "files": entries}
    model.save(tmp_path / "saved")
    (old_cache / "model.safetensors").unlink()
    calls = []

    def download(repo, filename, **kwargs):
        calls.append(filename)
        return str(new_cache / filename.split("/")[-1])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(FoundationRegressor, "_fit_context", lambda self: setattr(self, "model", object()))
    restored = FoundationRegressor.load(tmp_path / "saved")
    assert restored.checkpoint_reference_["path"] == str(new_cache)
    assert calls == ["regression/config.json", "regression/model.safetensors"]


@pytest.mark.parametrize("backend", ["nori", "tabldm", "tabfm"])
def test_constant_context_features_fail_before_upstream_allocation(fake_packages, monkeypatch, backend):
    def forbidden(self):
        pytest.fail("Invalid constant context must be rejected before loading model weights")

    monkeypatch.setattr(FoundationRegressor, "_fit_context", forbidden)
    with pytest.raises(ValueError, match="varying fingerprint feature"):
        FoundationRegressor(backend).fit(np.ones((4, 3)), np.arange(4))


@pytest.mark.parametrize("mask,device", [("7", "cuda:0"), ("0,0", "cuda:0"),
                                        ("GPU-uuid", "cuda:0"), ("0", "cuda:1"), ("", "cuda:0")])
def test_direct_api_rejects_forbidden_physical_gpu_mapping(monkeypatch, mask, device):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        FoundationRegressor("nori", device=device)
    # CPU callers never address an inherited visible GPU.
    FoundationRegressor("nori", device="cpu")


def test_runtime_rechecks_changed_device_mask_before_touching_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    model = FoundationRegressor("nori", device="cuda:0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        with model._runtime():
            pytest.fail("The invalid device must be rejected before CUDA initialization")


@pytest.mark.foundation
@pytest.mark.skipif(os.environ.get("ALD_RUN_FOUNDATION_TESTS") != "1",
                    reason="Set ALD_RUN_FOUNDATION_TESTS=1 to load the public pretrained models")
@pytest.mark.parametrize("backend", ["nori", "tabldm", "tabfm"])
def test_real_public_foundation_inference_and_restore(backend, tmp_path):
    """Optional network/CPU integration: actual packages and pretrained weights."""
    rng = np.random.default_rng(91)
    X = rng.integers(0, 2, size=(16, 8), dtype=np.uint8)
    y = -3 - X[:, 0] * 1.5 + X[:, 2] * .8 + rng.normal(0, .1, len(X))
    params = {"max_context_samples": 8, "max_features": 4}
    if backend != "nori":
        params.update(n_estimators=1, norm_methods=["none"])
    if backend == "tabldm":
        params["kv_cache"] = True
    model = FoundationRegressor(backend, params=params, cpu_threads=2, inference_batch_size=2)
    model.preflight()
    model.fit(PackedFingerprints.from_dense(X), y)
    mean, std = model.predict(PackedFingerprints.from_dense(X[:3]))
    assert mean.shape == std.shape == (3,)
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    assert np.all(std >= 0)
    if backend != "tabfm":
        assert np.any(std > 0)
    if backend == "tabldm":
        assert model.model.model_kv_cache_ is not None
    model.save(tmp_path)
    restored = FoundationRegressor.load(tmp_path)
    actual_mean, actual_std = restored.predict(X[:3])
    np.testing.assert_allclose(actual_mean, mean, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual_std, std, rtol=1e-6, atol=1e-7)
