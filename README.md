# Active Learning for Molecular Docking

A configuration-driven workflow that learns from acquired docking scores, predicts the remaining compound pool, and selects the next compounds within a fixed docking budget. Use **TabPFN v2** (the default), **TabM**, **Random Forest**, **Extra Trees**, **Synthefy Nori**, **Xiaomi TabLDM**, or **Google TabFM** individually or combine their regression predictions in an ensemble. Actual docking can optionally rescore each best pose with **GNINA CNNaffinity**.

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
| `docking` | AutoDock Vina or a configured QVina executable; optionally −CNNaffinity from GNINA rescoring | Prepare ligands and run iterative molecular docking |

[Quick start](#quick-start) · [Models and ensembles](#models-and-ensembles) · [Configuration](#configuration) · [Actual docking](#actual-docking) · [GNINA rescoring](#gnina-cnnaffinity-rescoring) · [Outputs](#outputs) · [Detailed reference](docs/reference.md)

> **Protocol change.** The default model is now **TabPFN v2** (previously TabM), and docking runs with `docking.rescoring: gnina` use **−CNNaffinity** (pK units) instead of the Vina/QVina energy (kcal/mol) as the score. Bundled historical results, [examples/verification.json](examples/verification.json), and earlier `results/`/`artifacts/` were produced with TabM on Vina-energy or synthetic scores. Do not mix them with new runs without stating which protocol each run used. An existing configuration that relied on the default model selects TabM again with `model.models: [tabm]` (or `--set 'model.models=[tabm]'`); without it, any TabM-only setting in that file is now a validation error. The bundled configurations no longer carry their former TabM hyperparameters, except `config_autotune.yaml`, which still selects TabM.

## Quick start

From a source checkout on **Linux x86_64 with glibc 2.28+**, run the command below. Bash, Git, GNU core utilities, `getconf`, and `tar` are required; `curl` or `wget` is needed only if uv must be downloaded. A preinstalled Python is not required.

```bash
bash install.sh
source .venv-ald/bin/activate

active-learning-docking models
active-learning-docking validate --config examples/replay.yaml
active-learning-docking run --config examples/replay.yaml
```

The installer uses **uv 0.11.7**, managed **Python 3.11.13**, and checked-in [CPU](requirements/install-cpu.lock) / [CUDA](requirements/install-cu128.lock) dependency locks. It installs all seven model backends, Vina/Meeko, receptor-preparation tools, tuning, and development tools into `.venv-ald`. The default PyTorch build is **2.8.0 CPU**. Installation includes real imports and a small TabM + Random Forest + Extra Trees replay to check that the environment runs. TabPFN (`tabpfn==9.2.0`) is a core dependency; an environment installed before TabPFN became the default lacks it and must be reinstalled (rerun `bash install.sh`, or use a new `--venv`) before running the default model. GNINA is not installed by `install.sh`; see [GNINA rescoring](#gnina-cnnaffinity-rescoring).

uv fits this repository's PyPI packages and pinned Git dependency, while supplying [managed Python installation](https://docs.astral.sh/uv/guides/install-python/) and [explicit PyTorch backend selection](https://docs.astral.sh/uv/guides/integration/pytorch/). If a matching uv is unavailable, the script downloads the official release archive, verifies its SHA-256, and caches the executable locally. It does not edit shell profiles or require conda or sudo.

To install and run the full checks, including actual pretrained TabPFN v2, Nori, TabLDM, and Google TabFM inference:

```bash
bash install.sh --test-models
```

Public checkpoints add approximately **7.2 GB** of downloads on first use, mostly Google TabFM. They are downloaded only when a pretrained model is executed; basic installation checks use TabM and the two forests. The default model's TabPFN v2 regression checkpoint is about **44 MB** and is downloaded from a pinned Hugging Face revision the first time a TabPFN run starts. All pretrained-model inference runs locally without API approval or tokens. Google TabFM's default weights are limited to [non-commercial, non-production use](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE).

### Installation options

| Command or option | Behavior |
|---|---|
| `bash install.sh` | Install every backend and tool with CPU PyTorch; check imports and a small core-model replay |
| `bash install.sh --test` | Also run the full test suite with native docking tests enabled |
| `bash install.sh --test-models` | Include `--test` and actual public-checkpoint tests for TabPFN v2, Nori, TabLDM, and Google TabFM |
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

Use Python 3.10+ for the core package, or Python 3.11+ for Google TabFM. The core package includes TabM, the forests, and TabPFN v2 (`tabpfn==9.2.0`). This path uses the package's dependency ranges rather than the installer's complete locks:

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

Choose models in the YAML `model.models` list. A one-item list runs only that backend; omitting the list selects **TabPFN v2** (`[tabpfn]`).

```yaml
model:
  models: [tabpfn]   # default; for example [tabm] or [tabpfn, random_forest, extra_trees]
```

`models: [tamb]` is accepted as an alias for `[tabm]`. A root-level `models` list is also accepted as shorthand, but do not specify it together with `model.models`. Canonical names are recorded in the resolved configuration.

| Name | How it learns | Package group |
|---|---|---|
| `tabpfn` | TabPFN v2 regression checkpoint; pretrained, conditioned on acquired examples without gradient training (default) | Core package |
| `tabm` (`tamb`) | Official TabM; trains its shared ensemble on acquired labels | Core package |
| `random_forest` (`rf`) | Random Forest regression | Core package |
| `extra_trees` (`et`) | Extra Trees regression | Core package |
| `nori` | Pretrained regression conditioned on acquired examples | `nori` extra |
| `tabldm` | Pretrained regression conditioned on acquired examples | `tabldm` extra |
| `tabfm` | Google TabFM, PyTorch regression backend | `tabfm` extra |

All rows above are installed by `bash install.sh`. The four pretrained (foundation) backends execute locally and use public pretrained weights; first use requires a download unless local checkpoints are configured. Nori uses `NoriRegressor`, which does not require a Synthefy API key.

**TabPFN v2 is the default.** The default model changed from TabM to TabPFN v2 (a lab decision recorded in the development notes). TabPFN v2 is a pretrained tabular model that predicts from the acquired examples supplied as context, without per-cycle gradient training, and returns a full predictive distribution whose standard deviation is used as the acquisition spread. TabM remains supported: select it with `models: [tabm]` (`config_autotune.yaml` does so because tuning is TabM-only). The [TabPFN v2 section of the model guide](docs/model_backends.md#tabpfn-v2-default) describes its parameters, checkpoint, and limits.

- **Spread has a different meaning.** TabPFN's spread is the exact standard deviation of its predictive distribution in score units, not the disagreement among TabM's ensemble members. The same `learning.beta` therefore explores differently with TabPFN than with TabM; choose `beta` for the selected model rather than reusing a TabM value.
- **CPU limit.** With `tabpfn` selected (alone or in an ensemble), `resources.device: cpu` and `learning.budget` greater than 1000 is a validation error, raised before any docking budget is spent. Use `resources.device: cuda:0` or `cuda:1` (physical GPU) for larger budgets. On physical GPU 1 with `resources.deterministic: true`, two identical TabPFN runs were byte-identical, so `examples/gpu.yaml` uses TabPFN.
- **TabM-only settings fail without TabM.** `n_blocks`, `d_block`, `k`, `dropout`, `learning_rate`, `weight_decay`, `epochs`, `patience`, `tuning_method`, and `n_trials` apply only to TabM. If `tabm` is not in `model.models` and any of them differs from its default, validation fails with a message that the value applies only to TabM. Remove the setting or add `tabm` to `models`. `batch_size`, `validation_fraction`, and `inference_batch_size` are shared by all models and are not checked. Google TabFM's default weights are restricted to **non-commercial, non-production use** under their [separate weight license](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE). See the [model guide](docs/model_backends.md) for sources, checkpoint paths, resource controls, and backend-specific options.

To combine models with the core installation:

```yaml
model:
  models: [tabpfn, random_forest, extra_trees]
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

Run the bundled [core ensemble](config_ensemble.yaml) (TabPFN + two forests) with `active-learning-docking run --config config_ensemble.yaml`. The [foundation example](examples/foundation.yaml) combines TabPFN with all three optional backends using small context/feature settings for the optional ones.

The default TabPFN v2 checkpoint is about 44 MB; among the optional backends, Nori-6M (about 47 MB) is the smallest download. TabFM's default checkpoint alone is about 6.59 GB; allow sufficient download time and memory. See [checkpoint sizes and installation](docs/model_backends.md#installation-and-public-checkpoints).

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
| [examples/replay.yaml](examples/replay.yaml) | Small synthetic replay with the default TabPFN; quickest execution check |
| [config_basic.yaml](config_basic.yaml) | Minimal synthetic replay (TabPFN) |
| [config_full.yaml](config_full.yaml) | Complete schema and settings |
| [config_ensemble.yaml](config_ensemble.yaml) | TabPFN + two forests with validation-derived weights; core installation |
| [examples/foundation.yaml](examples/foundation.yaml) | TabPFN + Nori + TabLDM + Google TabFM; optional extras |
| [config_autotune.yaml](config_autotune.yaml) | Seeded TabM hyperparameter search (selects `models: [tabm]`) |
| [examples/docking.yaml](examples/docking.yaml) | Actual Vina docking (TabPFN, no rescoring) |
| [examples/gpu.yaml](examples/gpu.yaml) | Physical GPU selection (TabPFN) |
| [examples/benchmark.yaml](examples/benchmark.yaml) | Paired acquisition comparisons (TabPFN) |

| Section | Main controls |
|---|---|
| `data` | CSV/store path, identity/SMILES/score columns, fingerprint type, radius, and bits |
| `model` | `models` (default `[tabpfn]`), `ensemble`, backend `params`, TabM-only training settings, shared batch sizes |
| `learning` | Initial sample size, query batch size, cycles, total budget, acquisition strategy |
| `evaluation` | Holdout fraction, random/scaffold split, top-K, prediction-table output |
| `resources` | CPU thread limit, physical GPU IDs, device, deterministic execution |
| `docking` | Engine, prepared receptor, box center/size, conformers, exhaustiveness, workers, optional GNINA rescoring (`rescoring`, `gnina_image`, `apptainer_executable`) |

All configurations use `schema_version: 1`. Unknown keys, duplicate YAML keys, invalid types, and contradictory resource settings fail. Relative paths, including `--set` values, resolve from the **YAML file's directory**. `validate` checks the schema. When a run starts, selected foundation runtimes and pretrained weights are checked before any docking budget is spent; the input chemistry and engine configuration are also checked.

`learning.initial_size` is queried at cycle 0. `learning.cycles` counts subsequent acquisition cycles; the total `learning.budget` includes initial queries and failed docking attempts. `learning.batch_size` counts compounds to dock, `model.batch_size` controls TabM training and is the fallback prediction batch for the other backends, and `model.inference_batch_size` controls prediction batches. TabPFN options live under `model.params.tabpfn`: `max_features` (500), `max_context_samples` (10000), `n_estimators` (8), and `model_path` (null = download the pinned checkpoint).

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

Set `evaluation.save_predictions: false` to avoid writing full-pool prediction tables. Tree and foundation backends materialize acquired training/context rows; their memory and inference costs differ from TabM. TabPFN pool prediction can be much slower than TabM's, so time a representative subset on the intended GPU before scaling the pool. The [large-library reference](docs/reference.md#large-libraries) records TabM/store measurements, storage estimates, and compatibility requirements.

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

### GNINA CNNaffinity rescoring

With `docking.rescoring: gnina` (docking mode only; the default is `none`), each compound's lowest-energy Vina/QVina pose is rescored by **GNINA 1.3.3** inside its isolated docking worker, and the training score becomes **−CNNaffinity**: GNINA's predicted affinity in pK units with the sign reversed, so lower remains better. It is not converted to kcal/mol and is not a measured affinity. The Vina/QVina engine still generates the poses; its energy is recorded but no longer trains the model. This follows the best single combination reported in *Integrating Machine-Learning-Based Pose Sampling with Established Scoring Functions for Virtual Screening* (J. Chem. Inf. Model. 2025, 65(10), 4833): Vina poses scored directly with `--cnn_scoring rescore`.

GNINA runs from the official GNINA 1.3.3 Docker image converted to an Apptainer SIF file and executed by rootless Apptainer; the official release binary needs glibc 2.35+ and cuDNN 9. Prepare it once before running:

```bash
# 1. Install Apptainer without root into a directory you own, using the official
#    tools/install-unprivileged.sh from the Apptainer release tag (see its -h for
#    distribution options). Apptainer 1.5.5 was used here.
bash tools/install-unprivileged.sh /path/to/apptainer
# 2. Keep the ~23.6 GB layer download and build files off the shared root disk.
export APPTAINER_CACHEDIR=/large/disk/apptainer/cache APPTAINER_TMPDIR=/large/disk/apptainer/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
# 3. Build the SIF (about 18.3 GB) from the official image, pinned by digest.
/path/to/apptainer/bin/apptainer build /large/disk/gnina/gnina-1.3.3.sif \
  docker://gnina/gnina@sha256:9da3dc61eef272cecb60798933a9a977d028c47024845fdf1f05a941ce5e25a2
```

Do not pull the 37.5 GB Docker image into a shared Docker store on the root disk. Then configure:

```yaml
docking:
  rescoring: gnina                                   # none (default) | gnina
  gnina_image: /large/disk/gnina/gnina-1.3.3.sif     # required with gnina; must be null otherwise
  apptainer_executable: /path/to/apptainer/bin/apptainer  # required with gnina; must be null otherwise
```

Both paths resolve relative to the YAML file; an executable name alone is not looked up on `PATH`. The receptor path and `output_dir` must not contain `:` or `,` (Apptainer bind syntax); both are rejected before docking starts.

- **Version check before budget is spent.** When the run creates the docking oracle, before any compound is docked, the Apptainer executable and SIF must exist and the image's `gnina --version` must report exactly one `gnina v1.3.3 …` line. Otherwise the run stops. The SIF's SHA-256 (about 20–40 s for an 18 GB file) and the version lines are written to the `rescoring` record of `manifest.json`. SIF files rebuilt from the same digest are not guaranteed to be byte-identical, so the version line, not the SIF hash, is checked.
- **Pose and input format.** The worker takes the one MODEL of the best conformer's pose file whose energy equals the recorded docking energy (the first such MODEL on ties) and writes it to `rescore_pose.pdbqt`. It then converts those coordinates, with bond orders and aromaticity from the compound's input SMILES, to a single-record `rescore_pose.sdf`, which GNINA scores. SDF is used because Meeko PDBQT files type some aromatic ring atoms as non-aromatic carbons, which changes GNINA's CNN input and its CNNaffinity.
- **Execution.** `--score_only --cnn_scoring rescore` with the GNINA 1.3.3 default CNN ensemble named explicitly (`dense_1_3 dense_1_3_PT_KD_3 crossdock_default2018_KD_4`), on CPU only (`--no_gpu`, no `--nv`, empty `CUDA_VISIBLE_DEVICES`), with `--cpu` equal to `docking.cpu`, a deterministic per-compound `--seed`, no network (`--net --network none`), and no host environment or home (`--containall --cleanenv --no-home`). Only the receptor (read-only) and the compound directory are bound. Rescoring took about 3.6 s per pose with the SIF on the development server, apparently dominated by reading libraries from the image.
- **Records.** Rescored runs add `docking_energy` (the Vina/QVina energy of that pose, kcal/mol), `cnn_score` (CNNscore), and `cnn_affinity` (raw CNNaffinity) after the existing columns of `observations.csv`, `final_results.csv`, and `cycle_*/queries.csv`, and to each compound's `result.json`. `score` equals `−cnn_affinity`, and `pose` points to `rescore_pose.sdf`. Each compound directory also keeps `rescore_pose.pdbqt`, `rescore_pose.sdf`, GNINA's output `gnina_scored.sdf` (values are read from its data items, which preserve float32 exactly), and `gnina.log`; `best_poses.pdbqt` is unchanged. With `rescoring: none`, compound records, pose files, and CSV columns are exactly as before; only the resolved configuration (and the manifest's copy of it) shows the three new keys.
- **Failures.** A GNINA nonzero exit, a missing or duplicated `CNNscore`/`CNNaffinity` item (GNINA omits an item whose value is exactly zero), a non-finite value, a failed pose selection or SDF conversion, or a timeout makes that compound a failed attempt: it consumes one budget unit, its score and all three extra columns stay empty (the docking energy is not substituted), and the reason is in `error` and `gnina.log`. `docking.timeout` covers preparation, docking of all conformers, SDF conversion, and GNINA together. A timeout kills the worker's whole session, including Apptainer helper processes.

GNINA scores from rescored runs are a different score definition from Vina/QVina energies; keep them separately identified. `scripts/build_replay.py` does not rescore; for a GNINA-based replay, supply a CSV score column that already contains −CNNaffinity. Details are in the [reference](docs/reference.md#gnina-cnnaffinity-rescoring).

## Outputs

Each run writes its resolved configuration, input/code/environment manifest, observations, ranked observed results, per-cycle metrics, model artifacts, and logs. Prediction tables are optional.

| Artifact | What to inspect |
|---|---|
| `config.resolved.yaml`, `manifest.json` | Effective configuration, versions, hashes, resource policy; `rescoring` record (GNINA image, SIF SHA-256, versions) for rescored runs |
| `observations.csv`, `final_results.csv` | Every attempted query and successful compounds ranked by score; rescored runs add `docking_energy`, `cnn_score`, `cnn_affinity` |
| `predictions.csv` | Final predicted mean/spread and queried/holdout status |
| `metrics.csv`, `summary.json` | Per-cycle and final performance/discovery summaries |
| `training_history.csv` | Recorded model fitting history |
| `model/`, `cycle_*/model/` | Fitted model artifacts and metadata |
| `cycle_*/training_split.csv` | Training/validation membership |
| `cycle_*/poses/` | Prepared ligands, engine records, and docking poses; GNINA inputs, output, and log for rescored runs |

For ensembles, model metadata records the selected backends and resolved weights. `member_training_splits.csv` records each member's training/validation/context membership; `ensemble_validation_split.csv` records the split used to learn weights. Context rows excluded by a configured cap are marked `unused`. [Artifact and checkpoint details](docs/model_backends.md#artifacts-and-reloading) and the [Python API](docs/reference.md#api-documentation) describe loading and reuse.

## Verification and development

Use `bash install.sh --test` for dependency/import checks, a small ensemble replay, and the suite with actual docking enabled. `bash install.sh --test-models` also executes the four public pretrained checkpoints (TabPFN v2, Nori, TabLDM, TabFM) and an exact-repeat, holdout-blind TabPFN replay. For focused work after activation:

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

Actual docking tests are opt-in: `ALD_RUN_DOCKING_TESTS=1 python -m pytest -q tests/test_docking.py -m integration`. The real GNINA rescoring test (repeat and one-versus-two-worker exactness) additionally needs `ALD_GNINA_IMAGE=<GNINA 1.3.3 SIF>` and `ALD_APPTAINER=<apptainer executable>`: `ALD_RUN_DOCKING_TESTS=1 ALD_GNINA_IMAGE=... ALD_APPTAINER=... python -m pytest -q tests/test_pipeline.py -m integration`. Without them it is skipped, and installer verification does not require it because the installer does not prepare GNINA. The default suite tests the rescoring contract with a fake Apptainer executable. Optional foundation tests and their required downloads are described in the [model guide](docs/model_backends.md#verification). Inspect skipped tests when assessing coverage.

The [detailed reference](docs/reference.md#validation-and-contributing-guidelines) preserves historical validation, the real-score fixture benchmark, and contributing invariants. Synthetic examples and small fixture benchmarks verify software behavior; they do not establish docking accuracy or superiority across targets.

Project licensing is in [LICENSE](LICENSE). Cite the selected upstream models, docking/preparation tools, structures, datasets, and this workflow's exact revision. [Citations and licenses](docs/reference.md#citation-and-license) (TabPFN v2, TabM, GNINA, and the rescoring protocol paper) and [other model sources](docs/model_backends.md#upstream-sources) provide the relevant references. The TabPFN v2 weights are under the Prior Labs License 1.1 (Apache 2.0 with an attribution requirement); the GNINA image carries the NVIDIA Deep Learning Container License notice.
