# Methods, API, and provenance reference

[Quick start](../README.md) · [Models and ensembles](model_backends.md)

This reference preserves the molecular-data rules, TabM training protocol, docking preparation details, Python API, and historical verification records. Commands and code paths below assume the repository root. Historical measurements apply to the named revisions and fixtures, not to every optional backend.

## Environment and reproducibility

The current supported installer is `bash install.sh` from a source checkout. It uses pinned uv 0.11.7 and managed Python 3.11.13 on Linux x86_64 with glibc 2.28 or newer. The default `.venv-ald` environment includes all six regression backends and the docking, preparation, tuning, and development tools. CPU PyTorch 2.8.0 is the default; `--torch-backend cu128` selects its CUDA 12.8 build. The installer consumes separate transitive dependency locks: [CPU](../requirements/install-cpu.lock) and [CUDA 12.8](../requirements/install-cu128.lock). [requirements/install.in](../requirements/install.in) defines their common inputs; each lock header records its regeneration command. These locks target that Python/Linux combination, rather than every platform allowed by the package metadata.

Use `--venv PATH` for another managed environment. The target path must be new, or an intact environment carrying this checkout's `.ald-install-owner` marker. Unmanaged environments, symlink targets, and environments owned by another checkout are rejected with an instruction to choose a new path. Rerunning an owned environment re-enforces locked versions and preserves unrelated packages. The installer uses its selected interpreter directly, so installation does not depend on the currently activated Python environment. `source activate_env.sh` activates `.venv-ald` by default and falls back to legacy `.venv`; an optional path argument selects a custom environment. `setup_server.sh` forwards all arguments to `install.sh`.

The shell installer requires Bash, Git, GNU core utilities, `getconf`, and `tar`. If uv 0.11.7 is absent from `PATH` and the installer cache, `curl` or `wget` downloads the official release archive; its embedded SHA-256 must match before extraction. The executable is cached at `${XDG_CACHE_HOME:-$HOME/.cache}/active-learning-docking/uv/0.11.7`. uv installs the managed Python without adding global Python links or changing shell profiles.

Basic installation checks package dependencies and native imports, then runs a three-model replay and verifies 12 successful acquisitions plus 48 finite pool predictions. `--test` enables the full suite with native docking tests, while `--test-models` additionally executes Nori, TabLDM, and Google TabFM using actual public pretrained weights. `--offline` requires the pinned uv, managed Python, and packages to be cached already; foundation execution also requires its cached checkpoints. See the [quick start and installer options](../README.md#quick-start) for commands.

Installation writes an `installed-freeze.txt` into the environment. Every verification uses a fresh `installation-checks/<timestamp-and-unique-suffix>/` directory containing `report.json`, `runtime.log`, the replay configuration/log/results, and optional `pytest.log`/`pytest.xml`. A failed verification records its error and traceback. Reports state the installed versions, resource settings, selected PyTorch backend, replay checks, and test/skip counts when the suite was requested; these per-run results establish what actually executed.

These installer locks are distinct from **`constraints-tested.txt`**, which records scientific versions from the older **Python 3.10** validation. That file is a historical constraint set, not the current all-backend environment or a complete platform-independent lock. To recreate that older setup manually, install the appropriate PyTorch 2.8.0 wheel first, then use `python -m pip install -c constraints-tested.txt -e '.[docking,tuning,dev]'` in a separate Python 3.10 environment. The historical measurements later in this document apply to their recorded revisions and environments.

PyTorch must support the requested CUDA device when GPU execution is selected. CPU is the default; requesting unavailable CUDA fails explicitly. The run manifest records the actual installed dependency versions, including PyTorch, RDKit, and model runtimes. Preserve that manifest, the appropriate installation lock, and an environment export for a reported experiment; dependency ranges alone do not specify an identical computational environment.

The Vina backend uses its Python package and needs no separate Vina binary. QVina requires an executable supplied through `docking.executable`; its file hash is recorded. Engine selection never silently falls back to a different implementation.

### Input data

A CSV file or a directory of CSV files is supported. Directory files are read in sorted order. Default identity and structure columns are `Catalog ID` and `smiles`; replay additionally requires the configured score column:

```csv
Catalog ID,smiles,docking_score
CMP001,CCO,-3.1
CMP002,c1ccccc1,-4.8
CMP003,CC(=O)Nc1ccccc1,-5.2
```

The displayed scores only illustrate the input format. Docking mode accepts the first two columns and obtains its own scores. Lower scores are always preferred, so transform any higher-is-better target explicitly before using it.

Identifiers remain strings. CSV headers must be unique, and every record must have exactly the header's number of fields. CSV syntax errors, duplicate headers, and extra or missing fields fail with the filename and logical record number, even with `data.invalid_policy: skip`. UTF-8 BOMs, quoted commas/newlines, and extra metadata columns are supported. Metadata columns are excluded from model inputs. `source_row` counts logical CSV records, with the header as record 1; embedded newlines do not increase it.

Structures are converted to canonical isomeric SMILES, and chirality-aware binary circular fingerprints provide the model inputs. `data.fingerprint_type` selects `morgan` (connectivity invariants, ECFP-like; default) or `feature_morgan` (pharmacophoric feature invariants, FCFP-like); both use `fingerprint_radius` (default 2) and `fingerprint_bits` (default 2,048). Fingerprints are stored as one `uint8` byte per bit, so a 2,048-bit matrix costs about 2 GB per million compounds; the model converts each batch to `float32` exactly. Neither scores nor compound identifiers enter featurization.

### Large libraries

A CSV input is validated, featurized, and held in memory, which is practical up to roughly 10^6 compounds. Larger pools, up to 10^8 compounds, use a **library store**: `prepare-library` streams the CSV files through `resources.cpu_threads` worker processes and writes an ID-sorted directory of NumPy columns (`compound_id.npy`, `smiles.npy`, `source.npy`, optional `scores.npy`), bit-packed `fingerprints.npy`, `rejected_rows.csv`, and `library.json` with the settings, input hashes, and per-file checksums. The store applies exactly the validation, canonicalization, deduplication, and representative rules of the CSV path, so a run from a store and a run from the same CSV produce identical scientific records. A store-backed run does not copy `library.csv` or `rejected_rows.csv` into the run directory; both live in the store, whose metadata and `library.json` hash are recorded in the manifest. Opening a store verifies every recorded file checksum, so a corrupted or partially copied store fails before any query. A failed build removes its partial outputs. Setting `data.path` to the store directory selects it; the configured fingerprint settings, `deduplicate`, `invalid_policy`, column names, and, for replay, `score_column` must match the store or the run fails before starting.

At run time the columns are memory-mapped and the model unpacks fingerprints one inference batch at a time, so the fingerprint matrix never materializes. Holdout splitting still parses every structure once (canonical SMILES or scaffolds), so use `holdout_fraction: 0` for very large stores; docking mode requires it anyway. The per-cycle full-pool prediction is the dominant cost; set `model.inference_batch_size` (for example 8192) and a CUDA device for large pools, keeping in mind that a different inference batch size can change floating-point rounding of validation losses and predictions in their last digits, and set `evaluation.save_predictions: false` to skip the per-cycle pool prediction tables and the final library-wide `predictions.csv`, which would otherwise be written for every compound. Replay mode still looks scores up by binary search on the sorted ID column. Resource requirements scale as follows:

| Quantity | Per compound | 10^8 compounds |
|---|---:|---:|
| `fingerprints.npy` on disk | `fingerprint_bits / 8` bytes | 12.8 GB at 1,024 bits; 25.6 GB at 2,048 bits |
| `smiles.npy` + `compound_id.npy` on disk | fixed width of the longest values | roughly 10–15 GB |
| Build RAM (IDs, digests, offsets, source positions) | about 40–60 bytes | 4–6 GB, plus the temporary files above on disk |
| Run RAM per adaptive cycle (masks, pool positions, predictions, spread, sort keys) | about 60 bytes | about 6 GB, plus PyTorch |

Measured on the development server with eight workers, the builder processed 2.0 million input records (895,315 accepted structures, the rest invalid or duplicate) in 1 min 43 s with a 1.3 GB peak, and a docking-mode cycle on that store predicted the full 895k pool on one L40 GPU in about 4 s with `inference_batch_size: 8192`. Linear extrapolation gives roughly 1.5 hours to build a 10^8 store and about 8 minutes of pool prediction per cycle; the sorted-order rewrite performs random reads over the temporary files, so an SSD is recommended. These are throughput measurements on a synthetic fragment library, not a scientific benchmark.

Malformed SMILES, missing identities, and nonfinite replay scores raise errors by default. `data.invalid_policy: skip` records rejected rows and removes them. Identical IDs are deduplicated; identical structures are also deduplicated by default, retaining the lexically smallest valid ID and that ID's original source row. Duplicate rejection records point to this final representative in `kept_compound_id`. Rearranging rows or files therefore does not change representative IDs or seeded selections. `data.deduplicate: false` retains distinct IDs for the same structure. Conflicting identities or conflicting scores for the same canonical structure always fail. This workflow does not enumerate protonation or tautomer states; prepare and document the chemical states appropriate to the scientific question.

The representative-ID rule can change selections for libraries with duplicate structures compared with earlier v2 runs. Existing experiment archives and bundled fixtures are retained as historical evidence and are not automatically regenerated.

### Configuration

For model selection and ensemble settings, see [the model guide](model_backends.md). All settings are versioned under `schema_version: 1`. Unknown keys, duplicate YAML keys, invalid types, and contradictory resource settings fail. Paths, including values supplied with `--set`, resolve relative to the configuration file. `config_full.yaml` contains every supported setting.

| Section | Principal settings and interpretation |
|---|---|
| Root | `seed`, `mode`, `output_dir` |
| `resources` | `cpu_threads` in 1–9, `gpu_ids` restricted to 0/1, `device`, `deterministic` |
| `data` | `path`, column names, `invalid_policy`, `deduplicate`, `fingerprint_type`, fingerprint radius/bits |
| `model` | `models`, `ensemble`, `params`; TabM settings: `n_blocks`, `d_block`, `k`, `dropout`, learning rate/weight decay, epochs, training batch size, `inference_batch_size`, patience, internal validation fraction, tuning |
| `learning` | `initial_size`, acquisition `batch_size`, `cycles`, total `budget`, `strategy`, `beta`, `beta_schedule`, `beta_final` |
| `evaluation` | `holdout_fraction`, `split` (`random` or `scaffold`), `top_k`, `save_predictions` |
| `docking` | Backend, prepared receptor, center/reference ligand, box dimensions, conformers, exhaustiveness, poses, timeout, CPU count per docking, concurrent `workers` |

`learning.batch_size` counts compounds queried per cycle; `model.batch_size` counts TabM training rows and is the fallback inference batch size. `model.inference_batch_size` controls pipeline prediction batches. Cycle 0 queries the initial random sample, and `learning.cycles` specifies subsequent acquisition cycles. The total attempted-compound budget includes the initial sample and all failed docking attempts. Selection stops at the first of budget exhaustion, candidate-pool exhaustion, or the configured cycle limit. A run needs at least three successful observations for model fitting. TabM additionally needs at least two distinct acquired molecular groups for its internal validation; validation-weighted ensembles reserve separate groups for weight selection as described in the [model guide](model_backends.md#ensemble-protocol).

CPU and GPU restrictions are applied before numerical-library initialization. Linux CPU affinity constrains the process and its children; native BLAS/OpenMP and PyTorch thread counts are bounded, tuning trials run sequentially, and docking runs at most `docking.workers` concurrent worker processes with `docking.cpu` threads each, where `workers × cpu` may not exceed `resources.cpu_threads`. `resources.device: cuda:1` names **physical GPU 1**, with logical remapping handled internally. Inherited numeric `CUDA_VISIBLE_DEVICES` restrictions are intersected, never widened. CPU runs hide all GPUs. Run separate experiments sequentially, or explicitly partition their CPU allocations when launching them concurrently.

### TabM training and acquisition methods

The package imports the official `tabm.TabM` architecture from the pinned `tabm==0.0.3` distribution. It does not vendor or redefine the architecture. TabM produces `k` predictions per compound using parameter sharing between ensemble members. This adapter uses the upstream `TabM.make` constructor with numerical fingerprint inputs and no numerical feature embeddings. For a standardized target `z`, training minimizes

```text
L = mean over compounds i and members j of (prediction[i,j] - z[i])²
```

At inference, predictions are averaged across members. This follows the [upstream training and inference conventions](https://github.com/yandex-research/tabm#training).

For the TabM backend, only acquired successful observations are passed to model fitting. Each cycle starts a fresh model with seed `experiment_seed + cycle`. An internal split groups identical canonical molecular structures together. Target mean and population standard deviation are calculated on the internal training rows alone, with unit scale for constant training targets. AdamW trains the model, validation monitors ensemble-mean MSE, and the checkpoint with the lowest internal validation error is restored. Validation and inference are batched. The training history states explicitly that losses are in standardized target units.

The uncertainty value is the population standard deviation across member predictions, converted back to target units. Members share parameters, so this spread is **not a calibrated confidence interval or a posterior standard deviation**. Dropout is disabled at inference. Calibration or uncertainty-driven scientific claims require separate assessment.

| Strategy | Selection rule for score minimization |
|---|---|
| `random` | Seeded uniform ordering of remaining compounds |
| `greedy` | Smallest ensemble mean |
| `lcb` | Smallest `mean - beta × spread` |
| `thompson` | Smallest independent normal draw per compound, with mean and `beta × spread` |

The Gaussian `thompson` rule is a heuristic, not a posterior sample. Libraries are sorted by compound ID, so positional ties equal lexical ID ties, and acquired compounds cannot be selected again.

`learning.beta` is used as is when `beta_schedule` is `constant`. A `linear` schedule moves the exploration weight from `beta` at the first adaptive cycle to `beta_final` at the configured last cycle, `β_c = β + (β_final − β)(c − 1)/(C − 1)`; an `exponential` schedule follows the geometric path `β_c = β (β_final/β)^{(c − 1)/(C − 1)}` and requires positive endpoints. The schedule is defined over the configured `cycles`, so a run that stops early for budget or pool exhaustion leaves later values unused. The effective beta of every adaptive cycle is recorded as `acquisition_beta` in the cycle metrics; cycle 0 is the random initial sample and records `null`. `greedy` and `random` ignore beta.

Optional TabM tuning uses the same acquired-set internal validation partition, with `n_jobs=1` and a seeded Optuna `RandomSampler` or `GridSampler`. The search domains are `n_blocks ∈ {1,2,3}`, `d_block ∈ {64,128,256}`, `dropout ∈ {0,0.1,0.2}`, `learning_rate ∈ {0.0005,0.002,0.005}`, and `weight_decay ∈ {0,0.0003}`. `k` and the training budget remain as configured. `n_trials` limits evaluations; a complete grid contains 162 combinations. The selected hyperparameters are applied to the final cycle model and recorded together with all trial results. Internal tuning scores are model-selection statistics, not unbiased test performance.

### Evaluation protocol

Replay separates an evaluation holdout before any query. Random holdouts group identical canonical molecules; scaffold holdouts group achiral Bemis–Murcko scaffolds, including one shared group for all acyclic molecules. Groups remain intact, so the realized holdout fraction can differ from the requested value. A requested holdout with fewer than two molecular/scaffold groups raises an error.

Holdout labels do not enter training, target scaling, tuning, early stopping, or acquisition. Holdout RMSE, MAE, R², and Spearman correlation are reporting outputs. Undefined metrics and metrics that overflow despite finite input scores, including discovery simple regret, are serialized as `null`. Do not use the reported holdout curves to choose hyperparameters and then describe that same partition as an untouched test set.

Discovery metrics use the eligible candidate pool only, excluding the holdout. For pool size `N`, reference top set size `K`, `m` acquired compounds, and `h` top-set hits:

```text
top-K recall = h / K
enrichment factor = (h / m) / (K / N)
simple regret = best acquired score - minimum candidate-pool score
```

`K` is capped by pool size, and ties use compound IDs. These replay metrics require exhaustive pool scores. Actual docking runs therefore require `evaluation.holdout_fraction: 0` and report observed scores and failures without implying an unobserved reference optimum.

## Examples

### Replay, repeated runs, and paired benchmarks

`examples/replay.csv` contains 48 molecular structures with a transparent synthetic descriptor score in arbitrary units. Its formula, generator version, and checksum are in `examples/replay.provenance.json`. Regenerate it with `python examples/make_demo.py`; the generated score is not a docking energy.

```bash
# Two fresh Python processes: exact comparison of scientific artifacts.
active-learning-docking reproduce --config examples/replay.yaml \
  --output results/reproducibility

# Paired splits and initial samples on the real Vina score fixture.
active-learning-docking benchmark --config examples/benchmark.yaml \
  --output results/benchmark --seeds 17 42 73 \
  --strategies random greedy lcb thompson

# Optional seeded hyperparameter search.
active-learning-docking run --config config_autotune.yaml

# Scaffold holdout example.
active-learning-docking run --config examples/replay.yaml \
  --set evaluation.split=scaffold --set output_dir=../results/scaffold

# Physical GPU 0; choose cuda:1 explicitly to use physical GPU 1.
active-learning-docking run --config examples/gpu.yaml
```

`reproduce` checks artifact integrity and compares query ordering, scores, predictions, uncertainties, split assignments, metrics, training histories, summaries, and pose bytes without a numerical tolerance. Both runs' per-cycle CSV and `metrics.json` inventories are compared; a file present in only one run fails comparison. CSV comparisons include headers, field order, and all records. Wall-clock timestamps and output-directory prefixes are excluded. It writes `reproducibility.json` and fails if any compared scientific record differs. Exact repeatability is assessed within the recorded software/hardware environment; different PyTorch/CUDA versions and devices are separate environments.

`benchmark` writes per-seed final metrics to `paired_results.csv`, with means and sample standard deviations in `benchmark.json`. Seeds are paired across strategies. The aggregate is descriptive; independent targets and libraries are needed to support broader conclusions.

### Actual molecular docking

```bash
active-learning-docking run --config examples/docking.yaml

# Repeat actual ligand preparation and docking in fresh processes.
active-learning-docking reproduce --config examples/docking.yaml \
  --output results/docking_reproducibility
```

The bundled example uses `inputs/candidates.csv`, the prepared `examples/data/receptor.pdbqt`, and a box centered on the heavy-atom centroid of `inputs/reference.pdb`. Its one-conformer, one-pose, exhaustiveness-1 settings provide a fast execution check. Choose scientifically appropriate search settings and validate the binding-site definition for production studies.

Ligand preparation uses seeded RDKit ETKDGv3 conformers, MMFF94 relaxation with UFF fallback when necessary, and Meeko PDBQT typing. Disconnected SMILES are rejected. Each conformer is docked, and the smallest finite docking energy is retained. All conformer inputs and poses are preserved. Per-compound seeds depend on the experiment seed, compound ID, and structure; they do not depend on evaluation order. Docking workers have an explicit CPU count and a timeout covering preparation and engine execution. On POSIX, timeout/interruption kills the worker's process group even if the worker has already exited; cleanup waits at most one additional second and preserves the original exception. Workers execute the absolute path of the caller's imported docking module.

`DockingOracle.evaluate` requires each compound's output directory to be new or empty. A nonempty directory or compound-directory link raises `FileExistsError` before writing or launching a worker. Successful results, failures, and interrupted attempts are all protected; use a new output location for retries. Internal docking also preflights all conformer, pose, engine-log, and best-pose targets and refuses existing files or links.

A reference ligand in PDB, PDBQT, SDF, MOL, or MOL2 format defines the box center as its heavy-atom centroid only when an explicit center is absent; multi-record SDF files use their first record. Successful scores are retained regardless of reference-pose RMSD, hydrogen-bond similarity, or score sign. Failed preparation/docking is logged and consumes budget; it never becomes an artificial numerical training label.

Each query is evaluated in its own isolated worker process. `docking.workers` greater than one evaluates the compounds of a cycle concurrently; per-compound seeds depend only on the experiment seed, compound ID, and structure, so the recorded observations, query order, budget accounting, and pose files do not depend on the worker count. Keep `workers × cpu` within `resources.cpu_threads` so the CPU affinity bound remains a hard ceiling.

The prepared example receptor has audited repairs to missing sidechain/terminal atoms and regenerated hydrogens at pH 7.4. Inspect `examples/data/receptor.provenance.json` for its source checksum, changed residues, software versions, and preparation command. The fixture does not substitute for documenting the receptor accession, chain selection, cofactors, chemical states, and binding-site rationale in a study.

To prepare another receptor using strict typing, activate the installer environment and run the command below. Receptor-preparation dependencies are already included by `bash install.sh`.

```bash
python scripts/prepare_receptor.py --input inputs/protein.pdb \
  --output results/preparation/receptor.pdbqt \
  --repair-missing-atoms --ph 7.4 --seed 42 --cpu 1
```

The explicit repair option rebuilds missing atoms and regenerates hydrogens; it does not reconstruct missing residues or silently remove unmatched residues. Omit it for already complete, appropriately hydrogenated structures. Review the generated provenance and preparation log. Both preparation and replay-generation scripts preflight every generated path, including prepared PDBs, logs, provenance, and artifact directories. Existing files or links (including dangling links), input aliases, and collisions between generated paths cause an argument error before any output is created. The [official Vina preparation tutorial](https://autodock-vina.readthedocs.io/en/latest/docking_basic.html) explains the required receptor/ligand preparation and docking-box inputs.

For QVina, set `docking.backend: qvina` and `docking.executable` to the executable path in a copied configuration. Use a separate output directory; scores from different engine settings should remain separately identified.

### Bundled molecular-system provenance

The receptor and ligand reference derive from the files already supplied in this
repository, `inputs/protein.pdb` and `inputs/reference.pdb`. Their experimental
identity, original structure preparation, and redistribution rights have not
been independently established by this refactor. The `COMPND` field of the
reference alone is insufficient to establish the provenance of the receptor.
These files are used to validate the software on an actual molecular system.

#### Receptor

Strict [Meeko receptor preparation](https://meeko.readthedocs.io/en/develop/cli_rec_prep.html)
initially rejected the supplied receptor because ten residues lack sidechain
heavy atoms. The example explicitly reconstructs those atoms with
[PDBFixer](https://github.com/openmm/pdbfixer) 1.12.0 and OpenMM 8.3.1, then applies
Meeko 0.7.1 with Gasteiger charges. No residues are discarded. Missing residues
and loops are not reconstructed. The repaired sidechains are modeled coordinates,
not experimentally observed coordinates.

The affected residues are A:100, A:107, A:155, A:158, A:161, A:272, A:298,
A:329, A:330, and A:343. The terminal oxygen OXT at A:382 is also added.
All 2,133 supplied heavy atoms are retained at their original coordinates;
39 heavy atoms are added. Input hydrogens are replaced using PDBFixer's default
residue pKa assignments at an assumed pH of 7.4. This is a declared example
preparation choice, not a target-specific protonation study.

`receptor.provenance.json` gives the complete atom list, command, package versions,
and input/output SHA-256 hashes. `receptor.prepared.pdb` preserves the intermediate
structure, and `receptor.pdbqt` is the frozen rigid receptor used for docking.
Preparation uses seed 42 and the OpenMM Reference CPU platform. An independent
repeat in the recorded environment produced identical PDBQT bytes.

After installing the `prepare` and `docking` extras, regenerate into a new path:

```bash
python scripts/prepare_receptor.py \
  --input inputs/protein.pdb \
  --output artifacts/receptor_repeat/receptor.pdbqt \
  --repair-missing-atoms --ph 7.4 --seed 42 --cpu 1
```

#### Ligands and replay labels

`docking_scores.csv` contains real AutoDock Vina 1.2.7 docking energies in kcal/mol
from attempted docking of the first 32 accepted unique structures in the supplied candidate CSV, in
original input order. Canonical isomeric SMILES deduplication precedes that fixed
selection; scores never influence membership. The disconnected salt Z8462127578 fails ligand preparation and contributes no numerical label; the fixture contains 31 successful scores and records the one failure. This small prefix is a software
fixture, not a representative chemical-library benchmark.

RDKit 2025.09.6 ETKDGv3 embedding and single-thread MMFF94 relaxation (UFF when
MMFF94 is unavailable) precede [Meeko ligand preparation](https://meeko.readthedocs.io/en/develop/lig_prep_basic.html).
Input SMILES chemical states and specified stereochemistry are retained. Salts
are rejected rather than silently stripped, and alternative protonation states,
tautomers, and unspecified stereoisomers are not enumerated. Docking uses
the [official Vina Python API](https://autodock-vina.readthedocs.io/en/latest/docking_python.html),
CPU 1, one conformer, exhaustiveness 1, one pose, and a 22 Å cubic box centered on
the reference ligand's heavy-atom centroid. Compound/conformer seeds derive from
SHA-256 of the configured seed and compound identity, so evaluation order does
not change their random streams.

`docking_scores.provenance.json` records every compound's score, seed, source row,
pose hash, engine versions, and receptor/input/configuration hashes. Generation
keeps all prepared ligands, docking poses, engine requests/results, and failure
records under `artifacts/reference_docking`. Failed calculations produce no
numerical label and are never replaced by a fabricated score.

To regenerate into separate paths:

```bash
python scripts/build_replay.py \
  --output artifacts/replay_repeat/docking_scores.csv \
  --artifacts artifacts/replay_repeat/docking \
  --count 32 --seed 42 --cpu 1 --exhaustiveness 1
```

The fixed-prefix regeneration intentionally exits with status 1 after writing complete provenance because the salt fails preparation. The expected result is 31 successful labels and one structured failure; inspect the report rather than treating it as 32 successful labels.

`scripts/build_replay.py` accepts a single input CSV file. Its output CSV may be inside the new artifact directory, outside the reserved `poses` tree and artifact filenames. Provenance records the hash of the docking module actually imported by the generator.

The deliberately inexpensive search settings support functional and repeatability
checks. Research use requires target-specific preparation review, suitable search
convergence studies, redocking/pose validation where applicable, and independent
experimental or appropriately designed retrospective evaluation. Docking energies
are not measured binding affinities.


## Outputs and provenance

```text
run/
├── config.resolved.yaml       # Complete effective configuration and absolute paths
├── manifest.json              # Inputs/code/artifact hashes, environment and resource policy
├── library.csv                # Canonical, accepted input records (CSV input; a store is referenced)
├── rejected_rows.csv          # Rejections and deduplication reasons (CSV input)
├── split.csv                  # Pool versus holdout identities (only when a holdout exists)
├── observations.csv           # Every attempted query, including structured failures
├── final_results.csv          # Successful observed compounds, ordered by score
├── predictions.csv            # Final mean/spread, partition and queried status (save_predictions)
├── metrics.csv                # Per-cycle performance/discovery summaries and acquisition_beta
├── training_history.csv       # Per-epoch member loss and validation ensemble loss
├── summary.json
├── active_learning.log
├── model/
│   ├── tabm_weights.pt          # Standalone TabM; other backends have their own artifacts
│   └── model_metadata.json
└── cycle_000/                  # Initial sample; cycle_001 begins adaptive acquisition
    ├── queries.csv
    ├── training_split.csv
    ├── metrics.json
    ├── model/
    └── poses/                 # Actual docking mode
```

Later cycles also save `pool_predictions_before_acquisition.csv` unless `evaluation.save_predictions` is false; replay cycles with a holdout save `holdout_predictions.csv`. Each docking compound directory records the request, result, worker log, prepared conformers, and poses. A failure marks the manifest as failed and preserves available diagnostics.

The manifest records input SHA-256 checksums, installed distributions, Git revision/dirty status when available, package source hashes, CPU affinity, thread settings, selected device, and deterministic-algorithm status. Archive the complete run directory with the exact dataset and code revision for a reported analysis.

## API documentation

### Package layout

| Module | Responsibility |
|---|---|
| `config.py` | Versioned dataclass schema, strict YAML loading, `--set` overrides, path resolution |
| `resources.py` | CPU affinity, thread limits, physical GPU masking, seeding; runs before numerical imports |
| `data.py` | CSV validation, canonicalization, deduplication, fingerprints, grouped holdout splits |
| `model.py` | `TabMRegressor`: official TabM training, tuning, checkpoints, ensemble mean/spread |
| `model_registry.py` | Canonical backend names, aliases, and installation hints |
| `regressors.py` | Model factory, tree regressors, cross-model ensembles, unified checkpoint loader |
| `foundation.py` | Local Nori, TabLDM, and Google TabFM adapters, context/feature selection, checkpoint context |
| `acquisition.py` | `random`, `greedy`, `lcb`, `thompson` selection with deterministic tie handling |
| `evaluation.py` | Regression and discovery metrics with JSON-safe undefined values |
| `docking.py` | Seeded ligand preparation, isolated Vina/QVina workers, `DockingOracle` |
| `store.py` | `PackedFingerprints`, `Library`, and the streaming `build_library_store` behind `prepare-library` |
| `pipeline.py` | `ActiveLearningExperiment` stages, `ReplayOracle`, `run_experiment` |
| `provenance.py` | Hashes, manifests, atomic JSON, output preflight |
| `experiments.py` | Fresh-process `reproduce` comparison and paired-seed `benchmark` |
| `cli.py` | `run`, `validate`, `models`, `reproduce`, `benchmark`, `prepare-library` commands |

`ActiveLearningExperiment` executes one configuration in stages: `prepare_library` (validation, featurization, sealed holdout), then per cycle `select_queries`, `query_oracle`, `fit_model`, and `record_metrics`, and finally `finalize`. The stages address compounds through `Library` (ID-sorted string columns, in memory for CSV input or memory-mapped for a store) and `PackedFingerprints` (bit-packed rows unpacked per batch), so the same code serves a 48-compound fixture and a 10^8-compound store. The oracle is any object with `evaluate(compound_id, smiles, output_dir)` returning `score`, `status` (`ok` or `failed`), `error`, and `pose`; `ReplayOracle` serves recorded scores and `DockingOracle` runs the docking engines. Exchanging the compound library, the receptor, the binding-site definition, or the score source therefore requires only configuration changes or a new oracle class, not pipeline edits.

Prefer the configuration API for full experiments:

```python
from active_learning_docking.config import load_config
from active_learning_docking.pipeline import run_experiment

config = load_config("examples/replay.yaml", ["output_dir=../results/python_api"])
summary = run_experiment(config)
```

`run_experiment` applies the resource policy itself unless a `runtime` from `configure_resources` is passed, as the CLI does before importing the pipeline. To evaluate compounds with a different score source, construct `ActiveLearningExperiment` with the same arguments and override `make_oracle` to return your oracle object; the remaining stages, records, and budget accounting are unchanged.

For direct model use, configure resources before importing numerical modules:

```python
from active_learning_docking.config import ResourceConfig
from active_learning_docking.resources import configure_resources

runtime = configure_resources(ResourceConfig(cpu_threads=2, device="cpu"), seed=42)

from active_learning_docking.data import featurize
from active_learning_docking.model import TabMRegressor

smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O"]
X = featurize(smiles, radius=2, n_bits=256)
y = [-3.1, -3.4, -4.8, -3.7]  # Illustrative scores only.
model = TabMRegressor(n_blocks=2, d_block=32, k=4, epochs=10,
                      seed=42, device=runtime["device"])
model.fit(X, y, groups=smiles)  # X is uint8; float32 input gives identical results.
mean, spread = model.predict(X)
members = model.predict_members(X)  # Shape: (number_of_compounds, k).
model.save("results/api_model")
restored = TabMRegressor.load("results/api_model", device=runtime["device"])
```

`fit` returns the estimator. `featurize` returns `uint8`; `fit`, `predict`, and `predict_members` accept boolean, integer, or floating-point matrices, or a lazy `PackedFingerprints` view (`PackedFingerprints.from_dense(X)` or a store's fingerprints with `.take(indices)`), and convert one `inference_batch_size` batch at a time. `groups` is optional for generic numeric use; molecular callers should supply canonical molecular identities. `predict` returns two vectors in target units; `predict_members` exposes all member outputs. `evaluate_all_metrics` uses the common regression metrics, and `score` returns R² or `None` when it is undefined. `history_`, `train_indices_`, `validation_indices_`, `best_epoch_`, `best_validation_loss_`, and `get_model_info()` expose the fitted protocol.

TabM checkpoints contain tensor weights and JSON metadata, including architecture, target normalization, validation membership, selected hyperparameters, history, and version information. Loading verifies the weights checksum and uses `torch.load(..., weights_only=True)`. It restores predictions; it is not an optimizer-state or interrupted-experiment resume facility.

### Migration from the earlier implementation

Version 2.0 intentionally changes the model, configuration, and pose-filtering protocol. `unified_model.UnifiedModel` remains an alias-compatible subclass of `TabMRegressor`, with `save_model`/`load_model` compatibility spellings. The old custom `TabMNetwork` MLP, inference-time Monte Carlo dropout, and their checkpoints are incompatible with the official TabM model. Retrain from source data. `mc_dropout_samples`, `embed_dim`, and model-level `max_threads` are removed; use `model.k`, `model.d_block`, and `resources.cpu_threads`.

The CLI now uses strict YAML plus `--set section.key=value`. Legacy pose-similarity filtering, MCS/RMSD training-label censoring, and hydrogen-bond thresholds are not part of the experiment pipeline. These changes alter the learning problem; do not combine earlier and current results without explicitly distinguishing their protocols.

## Validation and contributing guidelines

Run the suite within the same resource limits as experiments:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 \
  taskset -c 0,1 python -m pytest -q

ruff check active_learning_docking tests scripts main.py unified_model.py Activedocking.py
python -m build
```

On a scheduler-managed machine, replace `0,1` with two CPUs from the granted affinity set. Actual docking integration tests require both the optional engines and explicit `ALD_RUN_DOCKING_TESTS=1`: run `ALD_RUN_DOCKING_TESTS=1 python -m pytest -q tests/test_docking.py -m integration`. Inspect skip summaries when assessing coverage. The separate `reproduce` command executes fresh full experiments and verifies their scientific outputs.

Focused model checks compare the official output shape and member-loss objective, restore the actual best validation checkpoint, verify seeded training/tuning, test group separation and target-scaling isolation, and compare predictions after checkpoint loading. Data, acquisition, evaluation, resource, configuration, and docking checks exercise the surrounding protocol.

Keep changes scoped, make scientific assumptions explicit, and include behavior-level validation for methodological changes. Preserve the core invariants: acquired-label-only fitting, sealed evaluation holdouts, intact molecular groups, finite scores, honest failure accounting, deterministic query identities, and CPU/GPU restrictions. Update this README and the runnable configurations together when the interface changes. Do not interpret passing smoke tests or a small synthetic benchmark as evidence of a scientifically superior method.

### Historical verification (2026-09-06, before backend expansion)

The later large-library revision added `store.py`, the `prepare-library` command, the beta schedule, `inference_batch_size`, and `save_predictions`. **318 CPU tests** and the 3 Vina/QVina integration tests passed, Ruff passed, and builds succeeded. Store runs reproduced CSV runs exactly for replay and actual docking (observations, training splits, metrics, pose bytes), with one and two builder workers producing byte-identical stores; pre-revision runs differ only by the new `acquisition_beta` column. The synthetic 895k-compound store described under *Large libraries* ran a GPU docking experiment twice with identical records.

After the structural revision (staged `ActiveLearningExperiment`, oracle objects, `uint8` fingerprints, `feature_morgan`, `docking.workers`, SDF/MOL/MOL2 reference ligands, and removal of the unused legacy `exceptions` module and unittest files), **296 CPU tests and 3 actual Vina/QVina integration tests** passed with no failures. Every bundled configuration was rerun and compared with pre-revision runs using the `reproduce` comparison: CPU replay, scaffold/Thompson replay, autotune, physical GPU 1 replay, and the six-query actual-docking example with one and with two workers all matched exactly, including cycle metrics and pose bytes. A docking-mode `reproduce` with two workers passed, Ruff passed, and wheel/sdist builds succeeded. The paragraphs below describe the earlier passes of the same day.

Before this revision, **294 CPU tests and 3 actual Vina/QVina integration tests** passed, with no failures or skips in the selected suites. The 100 added regressions cover CSV structure and provenance, stable representatives, process cleanup, output preservation, module selection/hashing, symmetric archive comparisons, and JSON-safe regret. Independent CPU replay and six-query actual-docking runs matched exactly, including cycle metrics and pose bytes. Ruff passed, and wheel/sdist builds succeeded. Packaged modules and scripts were checked against the modified source. Evidence is under `artifacts/defect_fixes_20260906/`; existing fixtures were not regenerated.

Before these fixes, the complete suite passed **197 tests, with zero failures or skips**, in both the development environment (PyTorch 2.8.0+cu128) and a separately installed environment without system-site packages (PyTorch 2.8.0+cpu), using Python 3.10.12. This includes actual Vina/QVina integration and a two-conformer, three-pose Vina check. Ruff passed, wheel/sdist builds succeeded, the installed package ran from outside the repository, and the unpacked sdist ran its bundled example. The following broader experiment results describe that earlier verification pass.

Independent fresh-process repeats matched exactly for CPU replay, GPU 0 replay, GPU 1 replay, real-score replay, and a six-query actual-docking active-learning run. Receptor preparation was repeated independently. The 32-compound docking fixture was also regenerated independently: the 31 successful score records and pose files were byte-identical, and the disconnected-salt failure recurred without an invented score. Checkpoint reloads reproduced every saved mean/spread prediction for the main, shell-script, scaffold, and autotune examples. A holdout-label perturbation test verified that evaluation labels cannot change acquisition, training history, or model predictions.

The real-score integration benchmark used `examples/benchmark.yaml`, seeds 17/42/73, 31 scored compounds, 24 eligible pool compounds per seed, a budget of 16 queries including 8 initial queries, and top-K=4. Values below are means ± sample standard deviations across the three paired seeds:

| Policy | Top-4 recall | Enrichment factor | Holdout RMSE (kcal/mol) |
|---|---:|---:|---:|
| Random | 0.417 ± 0.520 | 0.625 ± 0.781 | 0.819 ± 0.044 |
| Greedy | 0.917 ± 0.144 | 1.375 ± 0.217 | 0.737 ± 0.102 |
| LCB | 0.750 ± 0.250 | 1.125 ± 0.375 | 0.757 ± 0.068 |
| Gaussian Thompson heuristic | 0.667 ± 0.144 | 1.000 ± 0.217 | 0.730 ± 0.068 |

This small, inexpensive, single-system fixture validates executable analysis; it does not establish statistical superiority, model calibration, or biological generalization. No hypothesis-test significance claim is made. These earlier measurements used dense fingerprints; the later library-store path keeps fingerprints bit-packed on disk and unpacks prediction batches. Missing-residue reconstruction, chemical-state enumeration, distributed execution, and interrupted-run resumption are outside this implementation.

The compact historical machine-readable record is [examples/verification.json](../examples/verification.json). Full local evidence is under `artifacts/`: JUnit reports in `verification/`, per-run manifests and exact-repeat reports, `real_benchmark/paired_results.csv`, `replay_reproducibility.json`, `docking_verification.json`, and `readme_validation.json`. Runtime artifacts are intentionally excluded from Git; regenerate and archive them with the commands above for your own analysis. All work here used CPU affinity within cores 0–7 (at most eight cores collectively) and physical GPUs 0/1 only.

## Citation and License

Cite the official TabM paper when using this model: Yury Gorishniy, Akim Kotelnikov, and Artem Babenko, *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*, ICLR 2025. [Paper](https://arxiv.org/abs/2410.24210), [official implementation](https://github.com/yandex-research/tabm).

```bibtex
@inproceedings{gorishniy2025tabm,
  title={TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling},
  author={Gorishniy, Yury and Kotelnikov, Akim and Babenko, Artem},
  booktitle={International Conference on Learning Representations},
  year={2025},
  url={https://arxiv.org/abs/2410.24210}
}
```

Also cite the docking engine, molecular preparation tools, source structures, and datasets used in the actual analysis. Record this workflow's exact revision and archived run artifacts. Project licensing is stated in [LICENSE](../LICENSE); upstream software and data retain their own licenses.
