# Models and ensembles

[Quick start](../README.md) · [Methods and provenance](reference.md)

All backends predict a continuous docking score from the same molecular fingerprints. The pipeline fits only successful acquired observations, then uses the predicted mean and spread to select more compounds. Selecting multiple backends changes model fitting and prediction; query budgets, docking failures, and sealed evaluation holdouts retain the same meaning.

## Selecting models

The canonical setting is a nonempty `model.models` list:

```yaml
model:
  models: [tabm]
```

The root shorthand `models: [tabm]` is also accepted. Specifying both locations is an error. Names are case-insensitive; aliases normalize to the canonical names below. Duplicate names, including aliases of the same backend, are rejected.

| Canonical name | Alias | Estimator | Additional runtime |
|---|---|---|---|
| `tabm` | `tamb` | Official `tabm.TabM` through `TabMRegressor` | None |
| `random_forest` | `rf` | scikit-learn `RandomForestRegressor` | None |
| `extra_trees` | `et` | scikit-learn `ExtraTreesRegressor` | None |
| `nori` | — | `synthefy_nori.NoriRegressor` | `nori` extra |
| `tabldm` | — | `tabldm.TabLDMRegressor` | `tabldm` extra |
| `tabfm` | — | Google `tabfm.TabFMRegressor`, PyTorch backend | `tabfm` extra, Python 3.11+ |

`active-learning-docking models` prints supported names and installation hints without loading the optional runtimes. Selecting one backend instantiates only that model. A missing optional runtime fails with an installation hint; a requested backend is never silently replaced.

Keep existing TabM training settings directly under `model`, for example `epochs`, `k`, `d_block`, and `learning_rate`. Optional `model.params.tabm` overrides those settings for TabM. Parameters for every other backend belong under `model.params.<canonical-name>`. Parameters for an unselected model are rejected. Resource and random-seed settings belong at experiment level; `device`, `seed`, `random_state`, `n_jobs`, and `cpu_threads` cannot override that policy through a backend parameter block.

## Installation and public checkpoints

The [one-command installer](../README.md#quick-start) installs all supported backends from checked-in dependency locks using managed Python 3.11.13:

```bash
bash install.sh
source .venv-ald/bin/activate
```

`bash install.sh --test-models` installs the same environment and additionally runs the full suite, native docking, and actual public-checkpoint tests for all three foundation models. The initial foundation downloads total approximately 7.2 GB; installing their runtimes alone does not download model weights. To select CUDA, use `--torch-backend cu128`; CPU PyTorch is the default. CUDA runtime packages add about 3.7 GiB (4 GB) of first-installation downloads, separately from model weights. The supported installer target is Linux x86_64 with glibc 2.28+.

The pinned default checkpoint downloads are approximately **47 MB for Nori-6M**, **569 MB for TabLDM**, and **6.59 GB for TabFM** (decimal sizes, excluding installed packages). Runtime memory also includes the loaded model and context. Public weights are cached and reused. For a small initial foundation run, select `models: [nori]`.

For a manual installation in your own environment, choose only the additional runtimes you need. Google TabFM requires Python 3.11+, and the TabLDM extra needs Git to install its pinned source revision:

```bash
python -m pip install -e '.[nori]'
python -m pip install -e '.[tabldm]'
python -m pip install -e '.[tabfm]'
```

Manual extras can be combined, for example `python -m pip install -e '.[docking,nori,tabldm,tabfm]'`. Core TabM/tree runs remain usable without foundation packages. A run resolves and hashes selected foundation checkpoints during preflight, before docking begins; downloads use pinned Hugging Face revisions. Preserve the resolved configuration, installed dependency versions, and checkpoint metadata when comparing experiments.

### Synthefy Nori

The adapter uses local `NoriRegressor`; it does not call Synthefy's hosted API. Public model names are `nori-6m`, `nori-30m`, and `nori-100m`. The public checkpoint downloads without an API key, Hugging Face token, or access request. A local checkpoint can be supplied through `model_path`. Nori's `fit` stores labeled context; prediction uses pretrained weights without task-specific gradient updates. [Official Nori documentation](https://github.com/Synthefy/synthefy-nori#quickstart), [public checkpoint access](https://github.com/Synthefy/synthefy-nori#authentication-optional).

### Xiaomi TabLDM

The adapter uses local `TabLDMRegressor`. Its pretrained regression checkpoint is fetched from Hugging Face on first use, or loaded from `model_path`. `fit` prepares context and loads weights; it does not optimize the model on docking data. Upstream exposes context caching and offloading controls for larger inputs. [Official installation and regression API](https://github.com/xiaomi-research/xiaomi-tabldm#regression).

### Google TabFM

This integration uses the **Google Research** TabFM PyTorch backend and regression weights from [`google/tabfm-1.0.0-pytorch`](https://huggingface.co/google/tabfm-1.0.0-pytorch). The public repository is accessible without a gated-model approval. Python 3.11+ is required. Upstream exposes sampled context/feature limits, `n_estimators`, and inference batching. [Official package and requirements](https://github.com/google-research/tabfm#installation).

The source code is Apache-2.0, but the default pretrained weights have a separate **non-commercial, non-production license**. It also restricts redistribution and commercial use of outputs; downloading without authentication does not remove these conditions. Keep the checkpoint in your local cache and record its reference/hash in shared experiment artifacts. Review the [weight license](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE) for the intended use.

The PyTorch regression weights occupy approximately **6.6 GB** on disk; plan for the download, cache, and runtime memory. Supply a local checkpoint **directory** containing both `config.json` and `model.safetensors`, or its parent with a `regression/` subdirectory. The supported loader does not load local `pytorch_model.bin` checkpoints. The project's `tabfm` extra includes the required `safetensors` runtime, and preflight checks it before downloading weights. [Loader source](https://github.com/google-research/tabfm/blob/main/tabfm/src/pytorch/tabfm_v1_0_0.py).

## Backend parameters

Foundation adapters default to **2,048 acquired context rows** and **128 fingerprint features**. Rows are sampled without replacement using the experiment seed; features are selected by variance on those context rows only, with deterministic ties. Query rows and evaluation labels do not fit this selection. These caps reduce inference cost but can discard informative fingerprint bits or acquired examples. Set either cap to `null` to disable it, and evaluate that choice on your task. `max_features` also controls TabFM's native feature cap; `null` disables both. Explicit upstream context or dimensionality options can still transform or subsample the input.

All three foundation backends require at least two acquired context rows and at least one fingerprint bit that varies within the selected context. Identical context fingerprints fail with an actionable error before model allocation; increase the context cap or acquire more diverse molecules.

After installing the optional extras, an ensemble with all requested foundation backends can be configured as follows:

```yaml
model:
  models: [tabm, nori, tabldm, tabfm]
  inference_batch_size: 128
  ensemble:
    strategy: mean
  params:
    nori:
      model: nori-6m
      max_features: 128
      max_context_samples: 2048
    tabldm:
      n_estimators: 8
      max_features: 128
      max_context_samples: 2048
    tabfm:
      n_estimators: 8
      max_features: 128
      max_context_samples: 2048
```

A complete small replay is provided in [examples/foundation.yaml](../examples/foundation.yaml); run `active-learning-docking run --config examples/foundation.yaml` from the project root. The [core ensemble configuration](../config_ensemble.yaml) runs without foundation packages.

Remove any names you do not want to run, and remove their corresponding `params` blocks. A single foundation model is selected with, for example, `model.models: [nori]`.

| Backend | Supported `model.params.<name>` options |
|---|---|
| All foundation backends | `max_features` (128), `max_context_samples` (2048); `null` disables a cap |
| `nori` | `model` (`nori-6m`), `model_path`, `augmentations`, `svd_dim` |
| `tabldm` | `model_path`, `allow_auto_download`, `checkpoint_version`, `n_estimators` (8), `batch_size`, `kv_cache`, `norm_methods`, `feat_shuffle_method`, `use_amp`, `offload_mode`, `disk_offload_dir` |
| `tabfm` | `checkpoint_path`, `n_estimators` (8), `batch_size`, `max_num_rows`, `norm_methods` |
| `random_forest`, `extra_trees` | Upstream constructor options such as `n_estimators` (128), `max_depth`, `min_samples_leaf`, `max_features`; `random_state`, `n_jobs`, and `warm_start` are controlled by the workflow |
| `tabm` | Existing TabM configuration fields; values here override the flat `model` fields |

Values in parentheses are adapter defaults. Other omitted options retain the pinned upstream runtime's defaults. In particular, explicitly setting TabFM's `max_num_rows` may further sample the acquired context after the adapter cap. A backend's `batch_size` controls its internal estimator batching; the outer `model.inference_batch_size` controls pool-row batches. See upstream sources for algorithm-specific choices.

TabFM 1.0.1's PyTorch regressor does not expose a context/KV cache. `cache_context`, `maybe_quantize_kv_cache`, and `keep_cache_on_device` are rejected before loading weights. Its loader caches the pretrained model within the process, but each prediction batch computes the context representations again. The loader uses upstream's default bfloat16 compute dtype on CPU and CUDA. Smaller context and estimator counts reduce inference work.

For checkpoints already on disk, set `model.params.nori.model_path`, `model.params.tabldm.model_path`, or `model.params.tabfm.checkpoint_path`. Relative paths resolve from the YAML file directory. TabLDM can disable downloads with `allow_auto_download: false`. Its `kv_cache` option accepts `false`, `true`, `kv`, or `repr`; the adapter forwards this into the upstream fit call so the requested cache is actually created. `offload_mode: disk` requires `disk_offload_dir`; that directory path also resolves from the YAML file. Feature shuffling accepts `none`, `shift`, `random`, or `latin`.

## Ensemble protocol

Start with a core ensemble:

```yaml
model:
  models: [tabm, random_forest, extra_trees]
  ensemble:
    strategy: mean
  params:
    random_forest:
      n_estimators: 128
    extra_trees:
      n_estimators: 128
```

Let model `m` produce mean `μ_m` and spread `σ_m`, and let normalized weights `w_m` sum to one. The combined prediction is:

```text
mean = Σ_m w_m μ_m
spread² = Σ_m w_m [σ_m² + (μ_m − mean)²]
```

Backend spread has the following meaning:

| Backend | Spread used for acquisition |
|---|---|
| TabM | Population standard deviation across shared ensemble members |
| Random Forest / Extra Trees | Population standard deviation across tree predictions |
| Nori / TabLDM | Approximate standard deviation derived from predicted quantiles |
| Google TabFM | Zero; its supported public regression API supplies point predictions |

For standalone TabFM, LCB and the Gaussian Thompson heuristic therefore select by predicted mean, as greedy does. In an ensemble, disagreement with other models can still produce nonzero spread. Do not interpret zero as certainty.

Each selected backend receives one model weight regardless of its internal member count. The variance includes within-model spread and disagreement between model means. It remains in docking-score units after taking the square root. It is not a calibrated confidence interval or a guarantee of improved predictions.

| Strategy | Weight calculation | Extra model fitting |
|---|---|---|
| `mean` | Equal model weights | None |
| `weighted` | Normalize user-supplied weights | None |
| `validation` | Normalize inverse validation MSE | Fit probes on a shared acquired-set split |

For explicit weights:

```yaml
model:
  models: [tabm, random_forest]
  ensemble:
    strategy: weighted
    weights: [0.7, 0.3]
```

Weights follow `models` order, must be finite and nonnegative, and need a positive sum. `weights` is valid only with `strategy: weighted`.

For automatic weighting:

```yaml
model:
  models: [tabm, random_forest]
  ensemble:
    strategy: validation
  validation_fraction: 0.2
```

Validation weighting uses one molecular-group split of the successful acquired labels. Each probe fits its training portion, predicts the same reserved validation rows, and receives a weight from its validation MSE. Any TabM tuning or early stopping remains inside the probe's training portion. The evaluation holdout is never used to fit or weight models. Final estimators are then fitted afresh from the acquired set with each backend's normal fitting protocol; TabM retains its own internal early-stopping split. The shared split (`ensemble_validation_split.csv`), validation scores, and resolved weights are recorded for review. MSE is floored at `1e-12` before inverse weighting so a perfect probe remains numerically well-defined. A one-model selection skips ensemble weighting entirely.

This validation score is a model-selection statistic. Use the sealed evaluation holdout for reported replay metrics, and keep model comparisons paired by seed, split, initial sample, and docking budget. Validation weighting adds work each cycle and may be unstable with very few acquired molecular groups.

## Resources and large pools

Use `resources.device` and `resources.cpu_threads` for execution control. CPU remains the default; CUDA IDs refer to physical GPUs 0 or 1. Tree estimators run on CPU even when neural backends use CUDA. Models are fitted sequentially within the experiment's resource limits.

`model.inference_batch_size` bounds the number of pool rows passed through each prediction call. A library store keeps pool fingerprints packed on disk; acquired rows still need memory for tree fitting and foundation context. Context length, fingerprint width, checkpoint size, and the number of backend estimators can make foundation inference substantially more expensive than TabM. Check timing and memory on a representative subset before increasing the pool size. Historical TabM throughput measurements do not describe the other backends.

## Artifacts and reloading

The resolved configuration records canonical model names and parameters. An ensemble writes `model_metadata.json` and a subdirectory named after each member, for example `model/tabm/` and `model/nori/`. Metadata includes model weights and validation MSE where applicable; these also appear in per-cycle metrics. `member_training_splits.csv` records each member's actual train/validation/context partition, with capped-out acquired rows marked `unused`. `ensemble_validation_split.csv` records weight-selection membership.

| Model | Saved state | Restore behavior |
|---|---|---|
| TabM | `tabm_weights.pt` and JSON metadata | Load tensor weights; existing API retained |
| Forest | `training_data.npz` and JSON metadata | Deterministically refit the acquired training data; original scikit-learn version required |
| Foundation | `foundation_context.npz` and JSON metadata | Recreate the estimator using saved context/features and upstream/local pretrained weights |
| Ensemble | Per-member directories and ensemble metadata | Restore members and the saved ensemble weights |

Numeric archives are loaded without pickle and checked against saved SHA-256 hashes. Foundation checkpoints preserve selected context and feature indices. Metadata records the upstream package version, Hugging Face repository/revision, resolved checkpoint paths, and file hashes. Loading checks the runtime version and checkpoint hashes before recreating the estimator. If a public checkpoint's original cache path is missing, the loader retrieves the saved immutable revision in the current cache and verifies the original hashes. Explicit local checkpoints must be restored at their recorded paths. Preserve the runtime environment and respect weight redistribution conditions when sharing artifacts. Restoration re-creates predictions; it does not resume an interrupted docking experiment.

Use the unified loader after configuring resources:

```python
from active_learning_docking.config import ResourceConfig
from active_learning_docking.resources import configure_resources

runtime = configure_resources(ResourceConfig(cpu_threads=2, device="cpu"), seed=42)

from active_learning_docking.regressors import load_regressor

model = load_regressor("results/run/model", device=runtime["device"])
# X must use the same fingerprint settings and column order as the run.
mean, spread = model.predict(X)
```

For direct construction, import `ModelConfig` from `active_learning_docking.config` and use `create_regressor(ModelConfig(models=["tabm", "extra_trees"]), seed=42, device=runtime["device"], cpu_threads=2)` from the same regressors module, then `fit(X, y, groups=canonical_smiles)`. A loaded experiment configuration can be passed as `create_regressor(config.model, seed=config.seed, device=runtime["device"], cpu_threads=config.resources.cpu_threads)`. The standalone [TabM Python API](reference.md#api-documentation), including `TabMRegressor.load`, remains available.

## Verification

The core suite includes adapter-contract tests using lightweight fake upstream classes. These verify routing, shapes, context selection, uncertainty, error handling, and persistence without downloading checkpoints.

For a complete installer-driven check, run `bash install.sh --test-models`. After activating the installed environment, individual public-checkpoint tests can also be selected directly:

```bash
# All three public foundation checkpoints: downloads missing weights.
ALD_RUN_FOUNDATION_TESTS=1 python -m pytest -q tests/test_foundation.py -m foundation

# Or verify one backend, for example the small Nori checkpoint.
ALD_RUN_FOUNDATION_TESTS=1 python -m pytest -q tests/test_foundation.py -m foundation -k nori

# Full active-learning loop with all four models.
active-learning-docking run --config examples/foundation.yaml
```

The installer saves its summary and detailed logs under `<environment>/installation-checks/<run>/`; inspect `report.json` and `pytest.xml` for the executed checks. Its `--test-models` mode requires successful Vina/QVina integration tests and all three real foundation tests, so skipped required tests cannot produce a successful verification.

The real tests fit acquired context, predict finite scores/spread, and compare saved/restored predictions. They require the optional packages, access to public weights or a populated cache, and sufficient memory. Without `ALD_RUN_FOUNDATION_TESTS=1`, these tests are explicitly skipped; inspect the skip summary when reporting coverage. Run manifests record dependency versions and model metadata records the exact checkpoint files/hashes. This verifies integration behavior, not docking prediction quality.

## Upstream sources

Use the citations and licensing instructions supplied by each model's maintainers:

- [TabM: official implementation and paper](https://github.com/yandex-research/tabm)
- [Synthefy Nori: official implementation, checkpoint links, and citation](https://github.com/Synthefy/synthefy-nori)
- [Xiaomi TabLDM: official implementation and technical report](https://github.com/xiaomi-research/xiaomi-tabldm)
- [Google TabFM: official implementation](https://github.com/google-research/tabfm) and [PyTorch model card](https://huggingface.co/google/tabfm-1.0.0-pytorch)

These links describe general tabular models. Their published benchmarks do not establish performance on this docking task.
