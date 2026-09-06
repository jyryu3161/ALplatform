"""Strict YAML schema and command-line override contract tests."""

import pytest

from active_learning_docking.cli import main
from active_learning_docking.config import from_dict, load_config


def test_relative_paths_resolve_against_config_not_working_directory(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("data:\n  path: candidates.csv\noutput_dir: output\n")
    monkeypatch.chdir(tmp_path.parent)
    config = load_config(path)
    assert config.data.path == str(tmp_path / "candidates.csv")
    assert config.output_dir == str(tmp_path / "output")


def test_overrides_set_default_values_and_false_booleans(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 73\ndata:\n  path: candidates.csv\nmodel:\n  auto_tune: true\nlearning:\n  beta: 3.0\n")
    config = load_config(path, ["seed=42", "model.auto_tune=false", "learning.beta=1.0", "learning.cycles=0"])
    assert config.seed == 42
    assert config.model.auto_tune is False
    assert config.learning.beta == 1.0
    assert config.learning.cycles == 0


@pytest.mark.parametrize("text", [
    "data:\n  path: a.csv\n  path: b.csv\n",
    "data:\n  path: a.csv\nunknown: 4\n",
    "data:\n  path: a.csv\nmodel:\n  typo: 4\n",
    "data:\n  path: a.csv\nseed: true\n",
    "data:\n  path: a.csv\nresources:\n  cpu_threads: 2.5\n",
    "data:\n  path: a.csv\nmodel:\n  auto_tune: 'false'\n",
    "data:\n  path: a.csv\nlearning:\n  beta: .nan\n",
    "data:\n  path: a.csv\ndocking:\n  center: [0, .inf, 0]\n",
    "[a, b]", "null", "", "3: value\n",
])
def test_invalid_yaml_schema_rejected(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("override", ["unknown=true", "model.typo=3", "seed", "=3", "model=false", "model.epochs=true"])
def test_invalid_overrides_rejected(tmp_path, override):
    path = tmp_path / "config.yaml"
    path.write_text("data:\n  path: a.csv\n")
    with pytest.raises(ValueError):
        load_config(path, [override])


@pytest.mark.parametrize("section,values", [
    ("resources", {"cpu_threads": 10}), ("resources", {"cpu_threads": 0}),
    ("resources", {"gpu_ids": [0, 2]}), ("resources", {"gpu_ids": [0, 0]}),
    ("resources", {"gpu_ids": [0], "device": "cuda:1"}),
    ("learning", {"initial_size": 20, "budget": 10}), ("learning", {"initial_size": 2}),
    ("evaluation", {"split": "not-a-split"}), ("evaluation", {"holdout_fraction": 0.5}),
    ("model", {"k": 1}), ("docking", {"center": [0, 0]}), ("docking", {"box_size": [1, 0, 1]}),
    ("docking", {"workers": 0}), ("docking", {"workers": 5, "cpu": 2}), ("data", {"fingerprint_type": "maccs"}),
    ("learning", {"beta_schedule": "cosine", "beta_final": 0.0}), ("learning", {"beta_final": 0.5}),
    ("learning", {"beta_schedule": "linear"}), ("learning", {"beta_schedule": "exponential", "beta_final": 0.0}),
    ("learning", {"beta_schedule": "linear", "beta_final": -1.0}), ("model", {"inference_batch_size": 0}),
])
def test_contradictory_configuration_rejected(section, values):
    with pytest.raises(ValueError):
        from_dict({"data": {"path": "a.csv"}, section: values})


def test_scheduled_beta_and_large_pool_keys_are_accepted():
    config = from_dict({"data": {"path": "a.csv"}, "learning": {"beta_schedule": "linear", "beta_final": 0.0},
                        "model": {"inference_batch_size": 8192}, "evaluation": {"save_predictions": False}})
    assert config.learning.beta_final == 0.0 and config.model.inference_batch_size == 8192
    assert config.evaluation.save_predictions is False


def test_docking_workers_share_the_cpu_budget():
    config = from_dict({"data": {"path": "a.csv"}, "resources": {"cpu_threads": 4}, "docking": {"workers": 2, "cpu": 2}})
    assert config.docking.workers * config.docking.cpu == config.resources.cpu_threads
    with pytest.raises(ValueError, match="workers"):
        from_dict({"data": {"path": "a.csv"}, "resources": {"cpu_threads": 3}, "docking": {"workers": 2, "cpu": 2}})


def test_live_mode_requires_prepared_receptor_center_and_no_holdout():
    with pytest.raises(ValueError, match="holdout_fraction"):
        from_dict({"data": {"path": "a.csv"}, "mode": "docking"})
    with pytest.raises(ValueError, match="receptor"):
        from_dict({"data": {"path": "a.csv"}, "mode": "docking", "evaluation": {"holdout_fraction": 0}})


def test_cli_validation_emits_resolved_config_and_honors_false(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("data:\n  path: a.csv\nmodel:\n  auto_tune: true\n")
    assert main(["validate", "--config", str(path), "--set", "model.auto_tune=false"]) == 0
    output = capsys.readouterr().out
    assert '"auto_tune": false' in output
    assert str(tmp_path / "a.csv") in output


def test_cli_invalid_config_exits_unsuccessfully(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("data:\n  path: a.csv\nunknown: 1\n")
    assert main(["validate", "--config", str(path)]) == 1
    assert "unknown configuration keys" in capsys.readouterr().err


def test_cli_malformed_yaml_exits_cleanly(tmp_path, capsys):
    path = tmp_path / "broken.yaml"
    path.write_text("data: [unclosed\n")
    assert main(["validate", "--config", str(path)]) == 1
    assert "Error:" in capsys.readouterr().err


@pytest.mark.parametrize("section,name,value", [
    ("learning", "beta", float("nan")), ("learning", "budget", True),
    ("resources", "gpu_ids", [False, True]), ("model", "auto_tune", "false"),
])
def test_directly_mutated_dataclasses_receive_same_strict_validation(section, name, value):
    config = from_dict({"data": {"path": "candidates.csv"}})
    setattr(getattr(config, section), name, value)
    with pytest.raises(ValueError):
        config.validate()


def test_model_selection_aliases_and_legacy_default():
    assert from_dict({"data": {"path": "x.csv"}}).model.models == ["tabm"]
    config = from_dict({"data": {"path": "x.csv"}, "models": ["tamb", "rf", "et"],
                        "model": {"params": {"rf": {"n_estimators": 8}},
                                  "ensemble": {"strategy": "weighted", "weights": [2, 1, 1]}}})
    assert config.model.models == ["tabm", "random_forest", "extra_trees"]
    assert config.model.params == {"random_forest": {"n_estimators": 8}}
    assert "models" not in config.to_dict()  # resolved schema has one canonical location
    with pytest.raises(ValueError, match="not both"):
        from_dict({"data": {"path": "x.csv"}, "models": ["tabm"], "model": {"models": ["tabm"]}})


@pytest.mark.parametrize("model", [
    {"models": []}, {"models": "tabm"}, {"models": ["tamb", "tabm"]}, {"models": ["typo"]},
    {"models": [True]}, {"models": ["random_forest"], "auto_tune": True},
    {"params": {"nori": {"model": "nori-6m"}}},
    {"params": {"tabm": {"seed": 20}}}, {"params": {"tabm": {"device": "cuda:2"}}},
    {"params": {"tabm": {"dropout": float("nan")}}}, {"params": {"tabm": []}},
    {"ensemble": {"strategy": "unknown"}}, {"ensemble": {"weights": [1]}},
    {"ensemble": {"strategy": "weighted"}},
    {"ensemble": {"strategy": "weighted", "weights": [0]}},
    {"ensemble": {"strategy": "weighted", "weights": [-1]}},
    {"ensemble": {"strategy": "weighted", "weights": [1, 2]}},
    {"ensemble": {"strategy": "weighted", "weights": [float("inf")]}},
    {"ensemble": {"strategy": "weighted", "weights": [True]}},
])
def test_invalid_model_selection_and_ensemble_fail_validation(model):
    with pytest.raises(ValueError):
        from_dict({"data": {"path": "x.csv"}, "model": model})


def test_optional_checkpoint_paths_and_model_overrides_resolve(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("data:\n  path: x.csv\n")
    config = load_config(path, ["model.models=[nori,tabfm]", "model.params.nori.model_path=weights/nori.pt",
                                "model.params.tabfm.checkpoint_path=weights/tabfm",
                                "model.params.nori.max_features=null"])
    assert config.model.params["nori"]["model_path"] == str(tmp_path / "weights/nori.pt")
    assert config.model.params["tabfm"]["checkpoint_path"] == str(tmp_path / "weights/tabfm")
    assert config.model.params["nori"]["max_features"] is None


def test_cli_lists_models_without_config_or_optional_imports(capsys):
    import json

    assert main(["models"]) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert {"tabm", "random_forest", "extra_trees", "nori", "tabldm", "tabfm"} == set(catalog["models"])
    assert catalog["aliases"]["tamb"] == "tabm"
