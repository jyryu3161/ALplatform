# Active Learning for Molecular Docking

A configuration-driven workflow that learns from acquired docking scores, predicts the remaining compound pool, and selects the next compounds within a fixed docking budget. Use **TabM**, **Random Forest**, **Extra Trees**, **Synthefy Nori**, **Xiaomi TabLDM**, or **Google TabFM** individually or combine their regression predictions in an ensemble.

```text
SMILES → molecular fingerprints → initial docking sample
                                      ↓
                     fit selected models → predict pool → select next batch
                                      ↑                         ↓
                                      └──── new docking scores ─┘
```

| Mode | Score source | Use |
|---|---|---|
| `replay` | A CSV containing known scores | Compare acquisition strategies with a held-out evaluation set |
| `docking` | AutoDock Vina or a configured QVina executable | Prepare ligands and run iterative molecular docking |

[Quick start](#quick-start) · [Models and ensembles](#models-and-ensembles) · [Configuration](#configuration) · [Actual docking](#actual-docking) · [Outputs](#outputs) · [Detailed reference](docs/reference.md)

## Quick start

From a source checkout on **Linux x86_64 with glibc 2.28+**, run the command below. Bash, Git, GNU core utilities, `getconf`, and `tar` are required; `curl` or `wget` is needed only if uv must be downloaded. A preinstalled Python is not required.

```bash
bash install.sh
source .venv-ald/bin/activate

active-learning-docking models
active-learning-docking validate --config examples/replay.yaml
active-learning-docking run --config examples/replay.yaml
```

The installer uses **uv 0.11.7**, managed **Python 3.11.13**, and checked-in [CPU](requirements/install-cpu.lock) / [CUDA](requirements/install-cu128.lock) dependency locks. It installs all six model backends, Vina/Meeko, receptor-preparation tools, tuning, and development tools into `.venv-ald`. The default PyTorch build is **2.8.0 CPU**. Installation includes real imports and a small TabM + Random Forest + Extra Trees replay to check that the environment runs.

uv fits this repository's PyPI packages and pinned Git dependency, while supplying [managed Python installation](https://docs.astral.sh/uv/guides/install-python/) and [explicit PyTorch backend selection](https://docs.astral.sh/uv/guides/integration/pytorch/). If a matching uv is unavailable, the script downloads the official release archive, verifies its SHA-256, and caches the executable locally. It does not edit shell profiles or require conda or sudo.

To install and run the full checks, including actual pretrained Nori, TabLDM, and Google TabFM inference:

```bash
bash install.sh --test-models
```

Public checkpoints add approximately **7.2 GB** of downloads on first use, mostly Google TabFM. They are downloaded only when a foundation model is executed; basic installation checks use the three core models. All foundation inference runs locally without API approval. Google TabFM's default weights are limited to [non-commercial, non-production use](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE).

### Installation options

| Command or option | Behavior |
|---|---|
| `bash install.sh` | Install every backend and tool with CPU PyTorch; check imports and a small core-model replay |
| `bash install.sh --test` | Also run the full test suite with native docking tests enabled |
| `bash install.sh --test-models` | Include `--test` and actual public-checkpoint tests for all three foundation models |
| `--torch-backend cu128` | Select PyTorch 2.8.0 with CUDA 12.8; requires a usable physical GPU 0 or 1 |
| `--venv PATH` | Use another installer-owned environment directory |
| `--offline` | Use cached uv, Python, and packages; foundation execution also needs cached weights |
| `--help` | Show available installer options |

For a separate CUDA environment:

```bash
bash install.sh --torch-backend cu128 --venv .venv-ald-cu128 --test
source .venv-ald-cu128/bin/activate
active-learning-docking run --config examples/gpu.yaml
```

CUDA runtime packages add about **3.7 GiB (4 GB)** of downloads on first installation, separately from the approximately 7.2 GB of foundation-model weights.

An existing environment at the requested path is reused only if this installer owns it for the same checkout; otherwise installation stops and requests a new `--venv` path. Reruns reapply the locked package versions and keep unrelated packages. `source activate_env.sh` selects `.venv-ald`, with `.venv` as a legacy fallback; use `source activate_env.sh PATH` for a custom environment. `bash setup_server.sh` forwards to the same installer. [Environment and reproducibility details](docs/reference.md#environment-and-reproducibility) describe lock files and the older validation environment.

Each verification creates `.venv-ald/installation-checks/<run>/report.json` with runtime/replay logs and, when requested, `pytest.log` and `pytest.xml`. The environment also contains `installed-freeze.txt`. With `--venv`, these files live beneath that chosen directory.

`python -m active_learning_docking` is an equivalent CLI entry point, and `python main.py --config config_basic.yaml` remains supported. Experiment output directories must be new or empty.

<details>
<summary>Manual pip installation in your own Python environment</summary>

Use Python 3.10+ for the core package, or Python 3.11+ for Google TabFM. This path uses the package's dependency ranges rather than the installer's complete locks:

```bash
python3.11 -m venv .venv-manual
source .venv-manual/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
# Select additional capabilities as needed:
python -m pip install -e '.[docking,tuning,dev,nori,tabldm,tabfm,prepare]'
```

| Extra | Installs |
|---|---|
| `docking` | AutoDock Vina 1.2.7, Meeko 0.7.1, and format support |
| `tuning` | Optuna for TabM tuning |
| `prepare` | PDBFixer/OpenMM and Meeko for receptor preparation |
| `nori` | Local Synthefy Nori runtime |
| `tabldm` | Official Xiaomi TabLDM runtime from a pinned Git revision |
| `tabfm` | Google TabFM with PyTorch and safetensors; Python 3.11+ |
| `dev` | Pytest, Ruff, and package-building tools |

All seven extras are included by `bash install.sh`. For a manual CUDA setup, install the appropriate PyTorch build before the package. Runtime device selection remains in `resources.device`; requesting unavailable CUDA fails explicitly.

</details>

## Models and ensembles

Choose models in the YAML `model.models` list. A one-item list runs only that backend; omitting the list keeps the original TabM behavior.

```yaml
model:
  models: [tabm]
```

`models: [tamb]` is accepted as an alias for `[tabm]`. A root-level `models` list is also accepted as shorthand, but do not specify it together with `model.models`. Canonical names are recorded in the resolved configuration.

| Name | How it learns | Package group |
|---|---|---|
| `tabm` (`tamb`) | Official TabM; trains its shared ensemble on acquired labels | Core package |
| `random_forest` (`rf`) | Random Forest regression | Core package |
| `extra_trees` (`et`) | Extra Trees regression | Core package |
| `nori` | Pretrained regression conditioned on acquired examples | `nori` extra |
| `tabldm` | Pretrained regression conditioned on acquired examples | `tabldm` extra |
| `tabfm` | Google TabFM, PyTorch regression backend | `tabfm` extra |

All rows above are installed by `bash install.sh`. The three foundation backends execute locally and use public pretrained weights; first use requires a download unless local checkpoints are configured. Nori uses `NoriRegressor`, which does not require a Synthefy API key. Google TabFM's default weights are restricted to **non-commercial, non-production use** under their [separate weight license](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE). See the [model guide](docs/model_backends.md) for sources, checkpoint paths, resource controls, and backend-specific options.

To combine models with the core installation:

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

| Ensemble strategy | Result |
|---|---|
| `mean` | Equal weight for each selected model; default |
| `weighted` | Normalize the supplied nonnegative `weights`, in `models` order |
| `validation` | Derive weights from inverse MSE on a shared validation split of acquired observations |

For explicit weights, replace the ensemble block with:

```yaml
  ensemble:
    strategy: weighted
    weights: [0.5, 0.25, 0.25]
```

Weights must match the model list and have a positive sum. Validation weighting never uses evaluation holdout labels; its extra fitting cost and exact protocol are described in the [model guide](docs/model_backends.md#ensemble-protocol). Combined uncertainty includes each backend's spread and disagreement between model means. It is an acquisition heuristic, not a calibrated confidence interval. More models do not guarantee better docking predictions; compare them on the same replay split and budget.

Run the bundled [core ensemble](config_ensemble.yaml) with `active-learning-docking run --config config_ensemble.yaml`. The [foundation example](examples/foundation.yaml) includes all three optional backends with small context/feature settings.

For a small foundation-model download, start with Nori-6M (about 47 MB). TabFM's default checkpoint alone is about 6.59 GB; allow sufficient download time and memory. See [checkpoint sizes and installation](docs/model_backends.md#installation-and-public-checkpoints).

TabFM's PyTorch backend has no context-cache option; context and estimator counts control its prediction cost. The [supported parameter table](docs/model_backends.md#backend-parameters) lists the accepted settings for each backend.

The CLI accepts the same model selection without editing YAML:

```bash
active-learning-docking run --config examples/replay.yaml \
  --set 'model.models=[tabm,random_forest]' \
  --set model.ensemble.strategy=validation \
  --set output_dir=../results/ensemble
```

## Configuration

Start with one of these files, change the input and output paths, and run `validate`:

| File | Purpose |
|---|---|
| [examples/replay.yaml](examples/replay.yaml) | Small synthetic replay; quickest execution check |
| [config_basic.yaml](config_basic.yaml) | Minimal synthetic replay |
| [config_full.yaml](config_full.yaml) | Complete schema and settings |
| [config_ensemble.yaml](config_ensemble.yaml) | TabM + two forests with validation-derived weights; core installation |
| [examples/foundation.yaml](examples/foundation.yaml) | TabM + Nori + TabLDM + Google TabFM; optional extras |
| [config_autotune.yaml](config_autotune.yaml) | Seeded TabM hyperparameter search |
| [examples/docking.yaml](examples/docking.yaml) | Actual Vina docking |
| [examples/gpu.yaml](examples/gpu.yaml) | Physical GPU selection |
| [examples/benchmark.yaml](examples/benchmark.yaml) | Paired acquisition comparisons |

| Section | Main controls |
|---|---|
| `data` | CSV/store path, identity/SMILES/score columns, fingerprint type, radius, and bits |
| `model` | `models`, `ensemble`, backend `params`, TabM training settings, inference batch size |
| `learning` | Initial sample size, query batch size, cycles, total budget, acquisition strategy |
| `evaluation` | Holdout fraction, random/scaffold split, top-K, prediction-table output |
| `resources` | CPU thread limit, physical GPU IDs, device, deterministic execution |
| `docking` | Engine, prepared receptor, box center/size, conformers, exhaustiveness, workers |

All configurations use `schema_version: 1`. Unknown keys, duplicate YAML keys, invalid types, and contradictory resource settings fail. Relative paths, including `--set` values, resolve from the **YAML file's directory**. `validate` checks the schema. When a run starts, selected foundation runtimes and pretrained weights are checked before any docking budget is spent; the input chemistry and engine configuration are also checked.

`learning.initial_size` is queried at cycle 0. `learning.cycles` counts subsequent acquisition cycles; the total `learning.budget` includes initial queries and failed docking attempts. `learning.batch_size` counts compounds to dock, `model.batch_size` controls TabM training, and `model.inference_batch_size` controls prediction batches.

Acquisition minimizes scores: `random` samples uniformly, `greedy` uses the smallest predicted mean, `lcb` uses `mean − beta × spread`, and `thompson` draws independent normal heuristic scores. Beta may stay constant or follow a linear/exponential schedule. [Protocol and evaluation details](docs/reference.md#tabm-training-and-acquisition-methods).

### Input data

Supply one CSV or a directory of CSVs. Default columns are:

```csv
Catalog ID,smiles,docking_score
CMP001,CCO,-3.1
CMP002,c1ccccc1,-4.8
CMP003,CC(=O)Nc1ccccc1,-5.2
```

These values illustrate the format. Docking mode needs only identity and SMILES columns. Lower scores are always preferred. Identifiers remain strings; metadata columns are excluded from model input. Canonical isomeric SMILES and chirality-aware Morgan or feature-Morgan fingerprints are used. Invalid rows fail by default; `data.invalid_policy: skip` records rejected rows. Duplicate structures have deterministic representative IDs. [Validation and deduplication rules](docs/reference.md#input-data).

### Large libraries

Build a reusable store to keep fingerprints bit-packed on disk and unpack only prediction batches:

```bash
active-learning-docking prepare-library --config examples/docking.yaml \
  --output results/candidate_store
active-learning-docking run --config examples/docking.yaml \
  --set data.path=../results/candidate_store \
  --set output_dir=../results/docking_store
```

Set `evaluation.save_predictions: false` to avoid writing full-pool prediction tables. Tree and foundation backends materialize acquired training/context rows; their memory and inference costs differ from TabM. The [large-library reference](docs/reference.md#large-libraries) records TabM/store measurements, storage estimates, and compatibility requirements.

## Actual docking

The default installer includes the docking tools. Run:

```bash
active-learning-docking run --config examples/docking.yaml
```

Vina uses its Python package. QVina requires `docking.backend: qvina` and the executable path in `docking.executable`. Configure a prepared receptor and either an explicit box center or a reference ligand. Docking mode requires `evaluation.holdout_fraction: 0`. Each attempt runs in an isolated worker, and `docking.workers × docking.cpu` must fit within `resources.cpu_threads`.

For explicit receptor repair and preparation (included by the installer):

```bash
python scripts/prepare_receptor.py --input inputs/protein.pdb \
  --output results/preparation/receptor.pdbqt \
  --repair-missing-atoms --ph 7.4 --seed 42 --cpu 1
```

The bundled receptor and fast docking settings are execution fixtures. Their [preparation, chemical-state assumptions, and provenance](docs/reference.md#bundled-molecular-system-provenance) should be reviewed before adapting the example to a scientific study. Docking failures consume budget and produce structured errors, never fabricated training scores. [Full docking workflow](docs/reference.md#actual-molecular-docking).

## Outputs

Each run writes its resolved configuration, input/code/environment manifest, observations, ranked observed results, per-cycle metrics, model artifacts, and logs. Prediction tables are optional.

| Artifact | What to inspect |
|---|---|
| `config.resolved.yaml`, `manifest.json` | Effective configuration, versions, hashes, resource policy |
| `observations.csv`, `final_results.csv` | Every attempted query and successful compounds ranked by score |
| `predictions.csv` | Final predicted mean/spread and queried/holdout status |
| `metrics.csv`, `summary.json` | Per-cycle and final performance/discovery summaries |
| `training_history.csv` | Recorded model fitting history |
| `model/`, `cycle_*/model/` | Fitted model artifacts and metadata |
| `cycle_*/training_split.csv` | Training/validation membership |
| `cycle_*/poses/` | Prepared ligands, engine records, and docking poses |

For ensembles, model metadata records the selected backends and resolved weights. `member_training_splits.csv` records each member's training/validation/context membership; `ensemble_validation_split.csv` records the split used to learn weights. Context rows excluded by a configured cap are marked `unused`. [Artifact and checkpoint details](docs/model_backends.md#artifacts-and-reloading) and the [Python API](docs/reference.md#api-documentation) describe loading and reuse.

## Verification and development

Use `bash install.sh --test` for dependency/import checks, a small ensemble replay, and the suite with actual docking enabled. `bash install.sh --test-models` also executes the three public foundation checkpoints. For focused work after activation:

```bash
# Compare scientific artifacts across fresh processes.
active-learning-docking reproduce --config examples/replay.yaml \
  --output results/reproducibility

# Paired seeds and initial samples across acquisition strategies.
active-learning-docking benchmark --config examples/benchmark.yaml \
  --output results/benchmark --seeds 17 42 73 \
  --strategies random greedy lcb thompson

# Unit/regression suite; tests select two CPUs from the allowed affinity set.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
  python -m pytest -q
ruff check active_learning_docking tests scripts main.py unified_model.py Activedocking.py
python -m build
```

Actual docking tests are opt-in: `ALD_RUN_DOCKING_TESTS=1 python -m pytest -q tests/test_docking.py -m integration`. Optional foundation tests and their required downloads are described in the [model guide](docs/model_backends.md#verification). Inspect skipped tests when assessing coverage.

The [detailed reference](docs/reference.md#validation-and-contributing-guidelines) preserves historical validation, the real-score fixture benchmark, and contributing invariants. Synthetic examples and small fixture benchmarks verify software behavior; they do not establish docking accuracy or superiority across targets.

Project licensing is in [LICENSE](LICENSE). Cite the selected upstream models, docking/preparation tools, structures, datasets, and this workflow's exact revision. [TabM citation and migration notes](docs/reference.md#citation-and-license) and [other model sources](docs/model_backends.md#upstream-sources) provide the relevant references.
