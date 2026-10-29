"""Strict, versioned experiment configuration with paths relative to the YAML file."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence, Union, get_args, get_origin, get_type_hints
import math

import yaml

from .model_registry import canonical_model_name


@dataclass
class ResourceConfig:
    cpu_threads: int = 8
    gpu_ids: list[int] = field(default_factory=lambda: [0, 1])
    device: str = "cpu"
    deterministic: bool = True


@dataclass
class DataConfig:
    path: str = ""
    id_column: str = "Catalog ID"
    smiles_column: str = "smiles"
    score_column: str = "docking_score"
    invalid_policy: str = "error"
    deduplicate: bool = True
    fingerprint_type: str = "morgan"
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048


@dataclass
class EnsembleConfig:
    strategy: str = "mean"
    weights: list[float] | None = None


# Fields read only by TabM. batch_size, validation_fraction and
# inference_batch_size are shared with other backends and the ensemble.
TABM_ONLY_FIELDS = ("n_blocks", "d_block", "k", "dropout", "learning_rate", "weight_decay", "epochs",
                    "patience", "tuning_method", "n_trials")


@dataclass
class ModelConfig:
    models: list[str] = field(default_factory=lambda: ["tabpfn"])
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_blocks: int = 3
    d_block: int = 128
    k: int = 16
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 0.0003
    epochs: int = 100
    batch_size: int = 128
    patience: int = 10
    validation_fraction: float = 0.2
    auto_tune: bool = False
    tuning_method: str = "random"
    n_trials: int = 10
    inference_batch_size: int | None = None

    def validate_selection(self) -> None:
        if not self.models:
            raise ValueError("model.models must contain at least one model")
        names = [canonical_model_name(name) for name in self.models]
        if len(names) != len(set(names)):
            raise ValueError("model.models must not contain duplicate models or aliases")
        parameters = {}
        for name, params in self.params.items():
            canonical = canonical_model_name(name)
            if canonical in parameters:
                raise ValueError("model.params contains duplicate model aliases")
            if canonical not in names:
                raise ValueError(f"model.params.{name} is not selected in model.models")
            forbidden = set(params) & {"device", "seed", "random_state", "n_jobs", "cpu_threads"}
            if forbidden:
                raise ValueError(f"Set resource/seed controls at experiment level, not model.params.{name}: "
                                 f"{sorted(forbidden)}")
            parameters[canonical] = params
        strategy, weights = self.ensemble.strategy, self.ensemble.weights
        if strategy not in {"mean", "weighted", "validation"}:
            raise ValueError("model.ensemble.strategy must be mean, weighted or validation")
        if strategy == "weighted":
            if (weights is None or len(weights) != len(names) or
                    any(type(w) not in {int, float} or not math.isfinite(w) or w < 0 for w in weights) or
                    max(weights) <= 0):
                raise ValueError("weighted ensemble requires one nonnegative weight per model, with positive sum")
        elif weights is not None:
            raise ValueError("model.ensemble.weights is only valid with strategy: weighted")
        if self.auto_tune and "tabm" not in names:
            raise ValueError("model.auto_tune requires tabm in model.models")
        if "tabm" not in names:
            # Stops an old TabM YAML from running silently on another model.
            for name in TABM_ONLY_FIELDS:
                value = getattr(self, name)
                if value != ModelConfig.__dataclass_fields__[name].default:
                    raise ValueError(f"model.{name}={value!r} applies only to TabM, which is not in "
                                     f"model.models {names}; remove it or add tabm to model.models")
        self.models, self.params = names, parameters


@dataclass
class LearningConfig:
    initial_size: int = 16
    batch_size: int = 8
    cycles: int = 5
    budget: int = 56
    strategy: str = "lcb"
    beta: float = 1.0
    beta_schedule: str = "constant"
    beta_final: float | None = None


@dataclass
class EvaluationConfig:
    holdout_fraction: float = 0.2
    split: str = "random"
    top_k: int = 10
    save_predictions: bool = True


@dataclass
class DockingConfig:
    backend: str = "vina"
    receptor: str | None = None
    reference_ligand: str | None = None
    center: list[float] | None = None
    box_size: list[float] = field(default_factory=lambda: [22.0, 22.0, 22.0])
    executable: str | None = None
    exhaustiveness: int = 8
    n_poses: int = 3
    conformers: int = 1
    timeout: float = 300.0
    cpu: int = 1
    workers: int = 1
    rescoring: str = "none"
    gnina_image: str | None = None
    apptainer_executable: str | None = None


@dataclass
class ExperimentConfig:
    schema_version: int = 1
    seed: int = 42
    mode: str = "replay"
    output_dir: str = "results/run"
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    docking: DockingConfig = field(default_factory=DockingConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return the nested plain-dictionary form used for YAML and manifests."""
        return asdict(self)

    def validate(self) -> ExperimentConfig:
        """Reject contradictory or unsafe experiments before importing numerical code."""
        _construct(ExperimentConfig, self.to_dict())
        self.model.validate_selection()
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not 0 <= self.seed < 2**31 - 1:
            raise ValueError("seed must be in [0, 2147483647)")
        if self.mode not in {"replay", "docking"}:
            raise ValueError("mode must be replay or docking")
        if not self.output_dir or not self.data.path:
            raise ValueError("output_dir and data.path are required")
        if not 1 <= self.resources.cpu_threads <= 9:
            raise ValueError("resources.cpu_threads must be between 1 and 9")
        ids = self.resources.gpu_ids
        if len(set(ids)) != len(ids) or not set(ids) <= {0, 1}:
            raise ValueError("resources.gpu_ids must be a unique subset of [0, 1]")
        if self.resources.device not in {"cpu", "cuda:0", "cuda:1"}:
            raise ValueError("resources.device must be cpu, cuda:0 or cuda:1 (physical GPU)")
        if self.resources.device != "cpu" and int(self.resources.device[-1]) not in ids:
            raise ValueError("requested device must be in resources.gpu_ids")
        if "tabpfn" in self.model.models and self.resources.device == "cpu" and self.learning.budget > 1000:
            raise ValueError("tabpfn on resources.device: cpu is limited to learning.budget <= 1000; "
                             "use resources.device: cuda:0 or cuda:1 (physical GPU) for larger budgets")
        for section, names in [
            (self.data, ["fingerprint_bits"]),
            (self.model, ["n_blocks", "d_block", "k", "epochs", "batch_size", "patience", "n_trials"]),
            (self.learning, ["initial_size", "batch_size", "budget"]),
            (self.evaluation, ["top_k"]),
            (self.docking, ["exhaustiveness", "n_poses", "conformers", "cpu", "workers"]),
        ]:
            for name in names:
                if getattr(section, name) < 1:
                    raise ValueError(f"{type(section).__name__}.{name} must be positive")
        if self.model.k < 2:
            raise ValueError("model.k must be at least 2 for ensemble uncertainty")
        if self.data.fingerprint_radius < 0 or self.learning.cycles < 0:
            raise ValueError("fingerprint_radius and cycles must be nonnegative")
        if self.data.fingerprint_type not in {"morgan", "feature_morgan"}:
            raise ValueError("data.fingerprint_type must be morgan or feature_morgan")
        if self.learning.initial_size > self.learning.budget:
            raise ValueError("learning.initial_size cannot exceed budget")
        if self.learning.initial_size < 3:
            raise ValueError("learning.initial_size must be at least 3")
        if self.data.invalid_policy not in {"error", "skip"}:
            raise ValueError("data.invalid_policy must be error or skip")
        if self.learning.strategy not in {"lcb", "greedy", "random", "thompson"}:
            raise ValueError("unknown learning.strategy")
        if self.learning.beta < 0:
            raise ValueError("learning.beta must be nonnegative")
        if self.learning.beta_schedule not in {"constant", "linear", "exponential"}:
            raise ValueError("learning.beta_schedule must be constant, linear or exponential")
        if self.learning.beta_schedule == "constant":
            if self.learning.beta_final is not None:
                raise ValueError("learning.beta_final requires a linear or exponential beta_schedule")
        else:
            if self.learning.beta_final is None or self.learning.beta_final < 0:
                raise ValueError("learning.beta_final must be a nonnegative number for a scheduled beta")
            if self.learning.beta_schedule == "exponential" and min(self.learning.beta, self.learning.beta_final) <= 0:
                raise ValueError("exponential beta_schedule requires positive beta and beta_final")
        if self.model.inference_batch_size is not None and self.model.inference_batch_size < 1:
            raise ValueError("model.inference_batch_size must be positive when set")
        if not 0 < self.model.validation_fraction < 0.5:
            raise ValueError("model.validation_fraction must be in (0, 0.5)")
        if not 0 <= self.model.dropout < 1:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.model.learning_rate <= 0 or self.model.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.model.tuning_method not in {"grid", "random"}:
            raise ValueError("model.tuning_method must be grid or random")
        if not 0 <= self.evaluation.holdout_fraction < 0.5:
            raise ValueError("evaluation.holdout_fraction must be in [0, 0.5)")
        if self.evaluation.split not in {"random", "scaffold"}:
            raise ValueError("evaluation.split must be random or scaffold")
        if self.mode == "docking" and self.evaluation.holdout_fraction != 0:
            raise ValueError("docking mode requires holdout_fraction: 0; use replay for held-out evaluation")
        if self.docking.backend not in {"vina", "qvina"}:
            raise ValueError("docking.backend must be vina or qvina")
        if self.docking.cpu > self.resources.cpu_threads:
            raise ValueError("docking.cpu cannot exceed resources.cpu_threads")
        if self.docking.workers * self.docking.cpu > self.resources.cpu_threads:
            raise ValueError("docking.workers × docking.cpu cannot exceed resources.cpu_threads")
        if self.docking.timeout <= 0:
            raise ValueError("docking.timeout must be positive")
        if self.docking.rescoring not in {"none", "gnina"}:
            raise ValueError("docking.rescoring must be none or gnina")
        rescoring_paths = (self.docking.gnina_image, self.docking.apptainer_executable)
        if self.docking.rescoring == "gnina":
            if self.mode != "docking":
                raise ValueError("docking.rescoring: gnina requires mode: docking")
            if not all(rescoring_paths):
                raise ValueError("docking.rescoring: gnina requires docking.gnina_image and "
                                 "docking.apptainer_executable")
            # Compound directories are bound into Apptainer; fail before any docking budget is spent.
            if any(character in str(self.output_dir) for character in ":,"):
                raise ValueError("docking.rescoring: gnina requires an output_dir without ':' or ','")
        elif any(value is not None for value in rescoring_paths):
            raise ValueError("docking.gnina_image and docking.apptainer_executable must be null "
                             "unless docking.rescoring is gnina")
        for name in ("box_size", "center"):
            values = getattr(self.docking, name)
            if values is not None and (len(values) != 3 or (name == "box_size" and min(values) <= 0)):
                raise ValueError(f"docking.{name} must contain three {'positive ' if name == 'box_size' else ''}numbers")
        if self.mode == "docking":
            if not self.docking.receptor:
                raise ValueError("docking.receptor is required (prepared PDBQT)")
            if self.docking.center is None and not self.docking.reference_ligand:
                raise ValueError("docking.center or reference_ligand is required")
            if self.docking.backend == "qvina" and not self.docking.executable:
                raise ValueError("docking.executable is required for qvina")
        return self


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[str, Any]:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("configuration keys must be strings")
        if key in mapping:
            raise ValueError(f"duplicate configuration key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _check_type(value: Any, annotation: Any, key: str) -> None:
    """Reject values whose runtime type differs from the dataclass annotation."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if args and (origin is Union or type(None) in args):
        if value is None and type(None) in args:
            return
        for candidate in args:
            try:
                _check_type(value, candidate, key)
                return
            except ValueError:
                pass
        raise ValueError(f"invalid type for {key}")
    if annotation is Any:
        if value is None or type(value) in {str, bool, int}:
            return
        if type(value) is float and math.isfinite(value):
            return
        if isinstance(value, list):
            for item in value:
                _check_type(item, Any, key)
            return
        if isinstance(value, dict) and all(isinstance(k, str) for k in value):
            for item in value.values():
                _check_type(item, Any, key)
            return
        raise ValueError(f"{key} must contain finite JSON-compatible values")
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be a mapping")
        for name, item in value.items():
            _check_type(name, args[0], key)
            _check_type(item, args[1], f"{key}.{name}")
    elif origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{key} must be a list")
        for item in value:
            _check_type(item, args[0], key)
    elif annotation is float:
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
    elif type(value) is not annotation:
        raise ValueError(f"{key} must be {annotation.__name__}")


def _construct(cls: type, mapping: Any, prefix: str = "") -> Any:
    """Build a dataclass from a mapping, rejecting unknown keys and wrong types."""
    if not isinstance(mapping, dict):
        raise ValueError(f"{prefix or 'configuration'} must be a mapping")
    hints = get_type_hints(cls)
    unknown = set(mapping) - set(hints)
    if unknown:
        raise ValueError(f"unknown configuration keys in {prefix or 'root'}: {sorted(unknown)}")
    values = {}
    for key, value in mapping.items():
        annotation = hints[key]
        if hasattr(annotation, "__dataclass_fields__"):
            values[key] = _construct(annotation, value, f"{prefix}{key}.")
        else:
            _check_type(value, annotation, f"{prefix}{key}")
            values[key] = value
    return cls(**values)


def from_dict(mapping: dict[str, Any], base_dir: str | Path | None = None) -> ExperimentConfig:
    """Build and validate a config; resolve input/output paths against base_dir."""
    if isinstance(mapping, dict) and "models" in mapping:
        mapping = dict(mapping)
        model = mapping.get("model", {})
        if not isinstance(model, dict) or "models" in model:
            raise ValueError("Use either root models or model.models, not both")
        mapping["model"] = dict(model, models=mapping.pop("models"))
    config = _construct(ExperimentConfig, mapping).validate()
    base = Path(base_dir or Path.cwd()).resolve()
    for obj, names in [(config, ["output_dir"]), (config.data, ["path"]),
                       (config.docking, ["receptor", "reference_ligand", "executable", "gnina_image",
                                         "apptainer_executable"])]:
        for name in names:
            value = getattr(obj, name)
            if value:
                setattr(obj, name, str((base / Path(value).expanduser()).resolve()))
    for params in config.model.params.values():
        for name in ("model_path", "checkpoint_path", "disk_offload_dir"):
            if params.get(name):
                if not isinstance(params[name], str):
                    raise ValueError(f"model.params paths ({name}) must be strings")
                params[name] = str((base / Path(params[name]).expanduser()).resolve())
    return config


def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> ExperimentConfig:
    """Load YAML with strict dotted-key overrides (values parsed as YAML scalars)."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        mapping = yaml.load(handle, Loader=_UniqueLoader)
    if not isinstance(mapping, dict):
        raise ValueError("configuration must be a nonempty mapping")
    for override in overrides or []:
        key, sep, raw = override.partition("=")
        if not sep or not key:
            raise ValueError("overrides must use section.key=value")
        current = mapping
        parts = key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
            if not isinstance(current, dict):
                raise ValueError(f"override parent for {key} is not a mapping")
        current[parts[-1]] = yaml.load(raw, Loader=_UniqueLoader)
    return from_dict(mapping, path.parent)
