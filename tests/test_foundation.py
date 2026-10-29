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


ORIGINAL_RESOLVE = FoundationRegressor._resolve_checkpoint

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


class FakeBarDistribution:
    """Z-space predictive distribution whose logits columns are (mean, variance)."""

    def variance(self, logits):
        return logits[:, 1]


TABPFN_SETTINGS = SimpleNamespace(settings=SimpleNamespace(tabpfn=SimpleNamespace(
    allow_cpu_large_dataset=True, max_batched_estimator_rows=7, max_batched_estimator_cells=9,
)))


class FakeTabPFNRegressor:
    """Keyword-only constructor like tabpfn 9.2.0; records the routed options."""

    def __init__(self, *, model_path, device, random_state, n_estimators, fit_mode, inference_precision,
                 memory_saving_mode, n_preprocessing_jobs, enable_torch_compile, ignore_pretraining_limits,
                 softmax_temperature, average_before_softmax):
        self.kwargs = {key: value for key, value in locals().items() if key != "self"}
        self.batches, self.fits = [], 0

    def fit(self, X, y):
        import torch

        settings = TABPFN_SETTINGS.settings.tabpfn
        self.fit_settings = (settings.allow_cpu_large_dataset, settings.max_batched_estimator_rows,
                             settings.max_batched_estimator_cells)
        self.fits += 1
        self.X, self.y = np.array(X), np.array(y)
        self.y_train_mean_, self.y_train_std_ = float(self.y.mean()), float(self.y.std()) + 1e-20
        self.is_constant_target_ = np.unique(self.y).size == 1
        self.znorm_space_bardist_ = FakeBarDistribution()
        self.distribution_dtype = torch.float32
        return self

    def predict(self, X, output_type="mean"):
        import torch

        assert output_type == "full"
        self.batches.append(np.array(X))
        z_mean = np.asarray(X, dtype=np.float64).sum(axis=1) / 10
        z_variance = 0.25 + np.asarray(X, dtype=np.float64)[:, 0]
        if self.is_constant_target_:
            return {"mean": np.full(len(X), self.y[0]), "logits": torch.zeros((len(X), 1))}
        logits = torch.as_tensor(np.c_[z_mean, z_variance], dtype=self.distribution_dtype)
        return {"mean": z_mean * self.y_train_std_ + self.y_train_mean_, "logits": logits}


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
        "tabpfn": SimpleNamespace(TabPFNRegressor=FakeTabPFNRegressor),
        "tabpfn.settings": TABPFN_SETTINGS,
        "tabpfn.model_loading": SimpleNamespace(download_model=lambda *args, **kwargs: "ok"),
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


@pytest.mark.parametrize("backend", ["nori", "tabldm", "tabfm", "tabpfn"])
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


def test_tabpfn_parameter_defaults_ranges_and_fixed_execution_options():
    model = FoundationRegressor("tabpfn")
    assert (model.max_features, model.max_context_samples) == (500, 10000)
    validate_foundation_params("tabpfn", {"max_features": 1, "max_context_samples": 2, "n_estimators": 1,
                                          "model_path": "weights/tabpfn-v2-regressor.ckpt"})
    validate_foundation_params("tabpfn", {"max_features": 500, "max_context_samples": 10000})


def test_tabpfn_null_model_path_is_the_pinned_download_default():
    validate_foundation_params("tabpfn", {"model_path": None})
    assert FoundationRegressor("tabpfn", params={"model_path": None}).params == {}
    with pytest.raises(ValueError):
        validate_foundation_params("nori", {"model_path": None})


@pytest.mark.parametrize("params", [
    {"max_features": 0}, {"max_features": 501}, {"max_features": None}, {"max_features": True},
    {"max_features": 2.0}, {"max_context_samples": 1}, {"max_context_samples": 10001},
    {"max_context_samples": None}, {"n_estimators": 0}, {"n_estimators": None}, {"n_estimators": "8"},
    {"model_path": ""}, {"model_path": 3},
    {"fit_mode": "fit_with_cache"}, {"inference_precision": "float64"}, {"memory_saving_mode": False},
    {"ignore_pretraining_limits": True}, {"device": "cpu"}, {"random_state": 1}, {"seed": 1},
    {"n_jobs": 1}, {"cpu_threads": 1}, {"checkpoint_path": "x"}, {"norm_methods": ["none"]},
])
def test_tabpfn_rejects_invalid_null_or_uncontrolled_parameters(params):
    with pytest.raises(ValueError):
        validate_foundation_params("tabpfn", params)


def test_tabpfn_uses_fixed_upstream_options_and_restores_pinned_settings(fake_packages):
    X = np.random.default_rng(3).integers(0, 2, (12, 6), dtype=np.uint8)
    model = FoundationRegressor("tabpfn", params={"n_estimators": 3}, seed=17).fit(X, np.arange(12.0))
    kwargs = model.model.kwargs
    assert kwargs.pop("model_path") == model.checkpoint_reference_["path"]
    assert kwargs == {"device": "cpu", "random_state": 17, "n_estimators": 3, **foundation._TABPFN_OPTIONS}
    assert model.model.fit_settings == (False, None, None)
    settings = TABPFN_SETTINGS.settings.tabpfn
    assert (settings.allow_cpu_large_dataset, settings.max_batched_estimator_rows,
            settings.max_batched_estimator_cells) == (True, 7, 9)
    info = model.get_model_info()
    assert info["implementation"] == "tabpfn.TabPFNRegressor (TabPFN v2 regression checkpoint)"
    assert info["fixed_upstream_options"] == {**foundation._TABPFN_OPTIONS,
                                              "pinned_settings": foundation._TABPFN_SETTINGS}
    assert info["uncertainty"] == "predictive_distribution_standard_deviation"
    assert FoundationRegressor("tabpfn").fit(X, np.arange(12.0)).model.kwargs["n_estimators"] == 8


def test_tabpfn_refuses_library_download_during_fit(fake_packages, monkeypatch):
    def fit_that_downloads(self, X, y):
        foundation.import_module("tabpfn.model_loading").download_model("anything")

    monkeypatch.setattr(FakeTabPFNRegressor, "fit", fit_that_downloads)
    loading = foundation.import_module("tabpfn.model_loading")
    original = loading.download_model
    with pytest.raises(RuntimeError, match="download"):
        FoundationRegressor("tabpfn").fit(np.eye(4), np.arange(4.0))
    assert loading.download_model is original


def test_tabpfn_selects_features_over_all_observations_before_seeded_sampling(fake_packages):
    rng = np.random.default_rng(5)
    X = rng.integers(0, 2, (40, 24), dtype=np.uint8)
    X[:, 3] = X[:, 7]  # identical columns: tie must keep the lower original index first
    packed = PackedFingerprints.from_dense(X)
    model = FoundationRegressor("tabpfn", params={"max_context_samples": 9, "max_features": 5},
                                seed=29).fit(packed, np.arange(40.0))
    counts = X.sum(axis=0).astype(np.int64)
    order = sorted(range(X.shape[1]), key=lambda j: (-(counts[j] * (len(X) - counts[j])), j))
    expected_features = np.sort(order[:5])
    expected_rows = np.sort(np.random.default_rng(29).choice(40, 9, replace=False))
    context_only = np.sort(np.argsort(-X[expected_rows].var(axis=0), kind="stable")[:5])
    assert not np.array_equal(expected_features, context_only)  # the ordering is observable
    np.testing.assert_equal(model.feature_indices_, expected_features)
    np.testing.assert_equal(model.train_indices_, expected_rows)
    np.testing.assert_equal(model.model.X, X[expected_rows][:, expected_features])
    np.testing.assert_equal(model.model.y, np.arange(40.0)[expected_rows])
    assert model.history_[0]["n_context"] == 9
    assert model.get_model_info()["feature_selection"] == "all_acquired_variance_before_context_sampling"


def test_tabpfn_variance_ties_follow_original_bit_index(fake_packages):
    # Same count (variance) at different row positions in every column.
    X = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
    model = FoundationRegressor("tabpfn", params={"max_features": 2}).fit(X, np.arange(4.0))
    np.testing.assert_equal(model.feature_indices_, [0, 3])


def test_tabpfn_context_sampling_is_seeded_and_only_below_the_limit(fake_packages):
    X = np.random.default_rng(8).integers(0, 2, (30, 10), dtype=np.uint8)
    full = FoundationRegressor("tabpfn", params={"max_context_samples": 30}).fit(X, np.arange(30.0))
    np.testing.assert_equal(full.train_indices_, np.arange(30))
    first = FoundationRegressor("tabpfn", params={"max_context_samples": 12}, seed=3).fit(X, np.arange(30.0))
    again = FoundationRegressor("tabpfn", params={"max_context_samples": 12}, seed=3).fit(X, np.arange(30.0))
    other = FoundationRegressor("tabpfn", params={"max_context_samples": 12}, seed=4).fit(X, np.arange(30.0))
    assert len(first.train_indices_) == 12
    np.testing.assert_equal(first.train_indices_, again.train_indices_)
    assert not np.array_equal(first.train_indices_, other.train_indices_)
    # Different context rows never change the all-observation feature selection.
    np.testing.assert_equal(first.feature_indices_, other.feature_indices_)


def test_tabpfn_queries_never_refit_or_reselect(fake_packages):
    rng = np.random.default_rng(12)
    X, pool = rng.integers(0, 2, (15, 8), dtype=np.uint8), rng.integers(0, 2, (50, 8), dtype=np.uint8)
    model = FoundationRegressor("tabpfn", params={"max_features": 4}, inference_batch_size=16)
    model.fit(X, np.arange(15.0))
    features, context = model.feature_indices_.copy(), model.model.X.copy()
    first = model.predict(pool)
    model.predict(1 - pool)
    second = model.predict(pool)
    assert model.model.fits == 1
    assert [len(batch) for batch in model.model.batches] == [16, 16, 16, 2] * 3
    np.testing.assert_equal(model.feature_indices_, features)
    np.testing.assert_equal(model.model.X, context)
    for before, after in zip(first, second):
        np.testing.assert_equal(before, after)


def test_tabpfn_spread_is_exact_predictive_distribution_std(fake_packages):
    X = np.random.default_rng(1).integers(0, 2, (10, 5), dtype=np.uint8)
    y = np.linspace(-9, -4, 10)
    model = FoundationRegressor("tabpfn").fit(X, y)
    queries = np.random.default_rng(2).integers(0, 2, (7, 5), dtype=np.uint8)
    mean, std = model.predict(queries)
    batch = queries[:, model.feature_indices_].astype(np.float64)
    z_variance = np.float32(0.25) + batch[:, 0].astype(np.float32)
    expected_std = np.sqrt(z_variance.astype(np.float64)) * model.model.y_train_std_
    np.testing.assert_array_equal(std, expected_std)
    expected_mean = batch.sum(axis=1) / 10 * model.model.y_train_std_ + model.model.y_train_mean_
    np.testing.assert_array_equal(mean, expected_mean)
    constant = FoundationRegressor("tabpfn").fit(X, np.full(10, -5.0))
    mean, std = constant.predict(queries)
    np.testing.assert_array_equal(mean, np.full(7, -5.0))
    np.testing.assert_array_equal(std, np.zeros(7))


@pytest.fixture
def tabpfn_weights(monkeypatch, tmp_path):
    """A stand-in checkpoint pinned by hash, plus a recording Hub downloader."""
    calls = []
    snapshot = tmp_path / "hub" / "snapshots" / foundation._TABPFN_CHECKPOINT["revision"]
    snapshot.mkdir(parents=True)
    weights = snapshot / "tabpfn-v2-regressor.ckpt"
    weights.write_bytes(b"test-only tabpfn v2 checkpoint")
    monkeypatch.setitem(foundation._TABPFN_CHECKPOINT, "sha256", FoundationRegressor._file_hash(weights))

    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(weights)

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        SimpleNamespace(hf_hub_download=download, snapshot_download=None))
    return weights, calls


def test_tabpfn_downloads_only_the_pinned_revision_without_token(fake_packages, monkeypatch, tabpfn_weights):
    weights, calls = tabpfn_weights
    monkeypatch.setattr(FoundationRegressor, "_resolve_checkpoint", ORIGINAL_RESOLVE)
    model = FoundationRegressor("tabpfn")
    model.preflight()
    assert calls == [("Prior-Labs/TabPFN-v2-reg", "tabpfn-v2-regressor.ckpt",
                      {"revision": "02a44705ff7790b50cded404bc59adb94da456a1", "token": False})]
    reference = model.checkpoint_reference_
    assert (reference["repository"], reference["revision"]) == ("Prior-Labs/TabPFN-v2-reg",
                                                                "02a44705ff7790b50cded404bc59adb94da456a1")
    assert reference["files"] == [{"path": str(weights), "sha256": foundation._TABPFN_CHECKPOINT["sha256"],
                                   "hub_filename": "tabpfn-v2-regressor.ckpt"}]
    model.fit(np.eye(4), np.arange(4.0))
    assert model.model.kwargs["model_path"] == str(weights)
    assert len(calls) == 1


def test_tabpfn_local_model_path_never_downloads(fake_packages, monkeypatch, tabpfn_weights, tmp_path):
    weights, calls = tabpfn_weights
    monkeypatch.setattr(FoundationRegressor, "_resolve_checkpoint", ORIGINAL_RESOLVE)
    local = tmp_path / "offline" / "tabpfn-v2-regressor.ckpt"
    local.parent.mkdir()
    local.write_bytes(weights.read_bytes())
    model = FoundationRegressor("tabpfn", params={"model_path": str(local)})
    model.preflight()
    model.fit(np.eye(4), np.arange(4.0))
    assert calls == []
    assert model.model.kwargs["model_path"] == str(local)
    assert model.checkpoint_reference_["repository"] is None


@pytest.mark.parametrize("name,content,error", [
    ("tabpfn-v2-regressor.ckpt", b"different weights", "SHA256"),
    ("tabpfn-v2.5-regressor-v2.5_default.ckpt", None, "tabpfn-v2-regressor.ckpt"),
])
def test_tabpfn_local_checkpoint_must_be_the_pinned_v2_file(fake_packages, monkeypatch, tabpfn_weights,
                                                             tmp_path, name, content, error):
    weights, calls = tabpfn_weights
    monkeypatch.setattr(FoundationRegressor, "_resolve_checkpoint", ORIGINAL_RESOLVE)
    local = tmp_path / "offline" / name
    local.parent.mkdir()
    local.write_bytes(content if content is not None else weights.read_bytes())
    model = FoundationRegressor("tabpfn", params={"model_path": str(local)})
    with pytest.raises(ValueError, match=error):
        model.preflight()
    assert model.checkpoint_reference_ is None and calls == []


@pytest.mark.parametrize("field,value,error", [
    ("revision", "main", "revision"), ("repository", "Prior-Labs/TabPFN-v2.5", "revision"),
    ("sha256", "0" * 64, "SHA256"), ("options", "fit_with_cache", "execution options"),
])
def test_tabpfn_restore_rejects_changed_provenance(fake_packages, monkeypatch, tabpfn_weights, tmp_path,
                                                   field, value, error):
    monkeypatch.setattr(FoundationRegressor, "_resolve_checkpoint", ORIGINAL_RESOLVE)
    X = np.random.default_rng(6).integers(0, 2, (9, 6), dtype=np.uint8)
    model = FoundationRegressor("tabpfn", params={"max_features": 3}).fit(X, np.arange(9.0))
    model.save(tmp_path / "saved")
    restored = FoundationRegressor.load(tmp_path / "saved")
    for actual, expected in zip(restored.predict(X), model.predict(X)):
        np.testing.assert_array_equal(actual, expected)
    path = tmp_path / "saved" / "model_metadata.json"
    metadata = json.loads(path.read_text())
    info = metadata["model_info"]
    if field == "options":
        info["fixed_upstream_options"]["fit_mode"] = value
    elif field == "sha256":
        info["checkpoint_reference"]["files"][0]["sha256"] = value
    else:
        info["checkpoint_reference"][field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=error):
        FoundationRegressor.load(tmp_path / "saved")


@pytest.mark.foundation
@pytest.mark.skipif(os.environ.get("ALD_RUN_FOUNDATION_TESTS") != "1",
                    reason="Set ALD_RUN_FOUNDATION_TESTS=1 to load the public pretrained models")
@pytest.mark.parametrize("backend", ["nori", "tabldm", "tabfm", "tabpfn"])
def test_real_public_foundation_inference_and_restore(backend, tmp_path):
    """Optional network/CPU integration: actual packages and pretrained weights."""
    rng = np.random.default_rng(91)
    X = rng.integers(0, 2, size=(16, 8), dtype=np.uint8)
    y = -3 - X[:, 0] * 1.5 + X[:, 2] * .8 + rng.normal(0, .1, len(X))
    params = {"max_context_samples": 8, "max_features": 4}
    if backend in {"tabldm", "tabfm"}:
        params.update(n_estimators=1, norm_methods=["none"])
    if backend == "tabpfn":
        params["n_estimators"] = 2
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
    if backend == "tabpfn":
        np.testing.assert_array_equal(actual_mean, mean)
        np.testing.assert_array_equal(actual_std, std)
        reference = restored.checkpoint_reference_
        assert (reference["repository"], reference["revision"]) == (
            "Prior-Labs/TabPFN-v2-reg", "02a44705ff7790b50cded404bc59adb94da456a1")
        assert reference["files"][0]["sha256"] == (
            "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736")
        # The library's own mean belongs to the same distribution as the exact spread.
        # Use the first prediction batch: batch composition can change the last digits.
        batch = np.asarray(X[:2], dtype=np.float32)[:, restored.feature_indices_]
        with restored._runtime():
            full = restored.model.predict(batch, output_type="full")
            decoded = restored.model.znorm_space_bardist_.mean(full["logits"]).cpu().numpy()
        scale, shift = restored.model.y_train_std_, restored.model.y_train_mean_
        np.testing.assert_array_equal(decoded.astype(np.float64) * scale + shift, actual_mean[:2])


@pytest.mark.foundation
@pytest.mark.skipif(os.environ.get("ALD_RUN_FOUNDATION_TESTS") != "1",
                    reason="Set ALD_RUN_FOUNDATION_TESTS=1 to load the public pretrained models")
def test_real_tabpfn_replay_runs_are_exact_and_holdout_blind(tmp_path):
    """Real TabPFN v2 replay: single and ensemble runs, exact repeat, blind to holdout labels."""
    import pandas as pd

    from active_learning_docking.config import TABM_ONLY_FIELDS
    from active_learning_docking.data import featurize
    from active_learning_docking.experiments import compare_runs
    from active_learning_docking.regressors import load_regressor
    from tests.test_pipeline import ROOT, run_cli, tiny_mapping

    def mapping_for(name, models, data_path=None):
        mapping = tiny_mapping(tmp_path / name, data_path)
        for field in TABM_ONLY_FIELDS:  # TabM-only values are rejected without tabm
            mapping["model"].pop(field, None)
        mapping["model"].update(models=models, inference_batch_size=16, params={
            "tabpfn": {"n_estimators": 2, "max_features": 64, "max_context_samples": 10},
            **({"random_forest": {"n_estimators": 5}} if "random_forest" in models else {}),
        })
        run_cli(mapping, tmp_path / f"{name}.yaml")
        return tmp_path / name

    first = mapping_for("single", ["tabpfn"])
    repeat = mapping_for("single_repeat", ["tabpfn"])
    assert compare_runs(first, repeat)["passed"]
    summary = json.loads((first / "summary.json").read_text())
    assert summary["models"] == ["tabpfn"] and summary["attempted"] == summary["successful"] == 13
    exported = pd.read_csv(first / "predictions.csv", float_precision="round_trip")
    assert (exported.uncertainty > 0).all()

    partitions = pd.read_csv(first / "split.csv")
    heldout = set(partitions.loc[partitions.partition == "holdout", "compound_id"])
    data = pd.read_csv(ROOT / "examples/replay.csv", float_precision="round_trip")
    data.loc[data["Catalog ID"].isin(heldout), "synthetic_score"] -= 1000
    source = tmp_path / "changed_holdout.csv"
    data.to_csv(source, index=False)
    changed = mapping_for("changed_holdout", ["tabpfn"], source)
    for name in ("observations.csv", "predictions.csv", "training_history.csv", "split.csv"):
        assert (first / name).read_bytes() == (changed / name).read_bytes(), name
    for cycle in first.glob("cycle_*"):
        metadata = cycle / "model" / "model_metadata.json"
        counterpart = changed / cycle.relative_to(first) / "model" / "model_metadata.json"
        assert metadata.read_bytes() == counterpart.read_bytes()
    assert not pd.read_csv(first / "metrics.csv").holdout_rmse.equals(
        pd.read_csv(changed / "metrics.csv").holdout_rmse)

    ensemble = mapping_for("ensemble", ["tabpfn", "random_forest"])
    summary = json.loads((ensemble / "summary.json").read_text())
    assert summary["models"] == ["tabpfn", "random_forest"]
    assert summary["attempted"] == summary["successful"] == 13
    exported = pd.read_csv(ensemble / "predictions.csv", float_precision="round_trip")
    restored = load_regressor(ensemble / "model")
    mean, std = restored.predict(featurize(exported.smiles, n_bits=128))
    np.testing.assert_array_equal(mean, exported.prediction.to_numpy())
    np.testing.assert_array_equal(std, exported.uncertainty.to_numpy())
