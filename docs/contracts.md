# 외부 인터페이스 계약

소비자는 (1) CLI 사용자, (2) Python API를 쓰는 스크립트, (3) 실행 산출물을 읽는 분석 코드다. 인증·접근 제어는 없다(로컬 실행).

## 공통 규약

- **점수는 낮을수록 좋다.** 높을수록 좋은 목표는 입력 전에 부호를 바꾼다.
- **현재 모든 명령의 출력 위치는 새것이거나 비어 있어야 한다.** 아니면 아무것도 쓰기 전에 실패한다(`FileExistsError` 또는 인자 오류).
- **설정값의 상대 경로는 YAML 파일 디렉터리 기준**이다(`--set` 값 포함). 반면 CLI의 `--config`와 `--output`(reproduce, benchmark, prepare-library)은 **현재 작업 디렉터리 기준**이다. 저장소 루트에서 `--output results/x`와 `--set output_dir=../results/x`(설정 파일이 `examples/`에 있을 때)가 같은 `results/` 아래를 가리키는 이유다.
- `docking.executable`도 설정 경로로 해석되므로 `qvina2.1`처럼 이름만 주면 PATH에서 찾지 않고 YAML 디렉터리 기준 경로가 된다. 실행 파일 경로를 명시한다.
- **설정은 엄격하다**: 알 수 없는 키, 중복 키, 잘못된 타입, 모순된 값은 실패.
- **정의되지 않은 수치는 JSON `null`**이며 NaN/Infinity는 쓰지 않는다.
- **화합물 ID는 항상 문자열**이다(앞뒤 공백 제거).

## CLI: `active-learning-docking <command>`

(`python -m active_learning_docking`, `python main.py`와 동일)

| 명령 | 입력 | 출력 (stdout JSON) | 오류 |
|---|---|---|---|
| `models` | 없음 | `{"models": {이름: {implementation, install, access}}, "aliases": {...}}` | 없음 |
| `validate --config F [--set K=V]...` | YAML | 해석·검증된 전체 설정(절대 경로) | 스키마/값 오류 → 종료 1 |
| `run --config F [--set ...]` | YAML | 실험 `summary.json` 내용 | 종료 1 (아래) |
| `reproduce --config F --output D [--set ...]` | YAML, 새 D | `{passed, comparison: "exact", checks: {파일명: bool}, seed, mode, device}` | 불일치 → `RuntimeError` 종료 1 |
| `benchmark --config F --output D [--seeds 17 42 73] [--strategies random lcb]` | replay YAML, 새 D | `{seeds, strategies, aggregate: {전략: {지표: {mean, sample_std, n_seeds}}}, interpretation}`; D에 `paired_results.csv`, `benchmark.json` | 도킹 모드, 서로 다른 시드 2개 미만, 전략 중복·빈 목록 → 종료 1 |
| `prepare-library --config F --output D [--chunk-size 20000]` | YAML의 `data.*`, 새 D | `{n_compounds, n_rejected, n_input_records, fingerprint, files}` | 검증 실패 시 부분 출력 삭제 후 종료 1 |

- 오류 계약: `ValueError`, `OSError`, `RuntimeError`, `ImportError`, YAML 오류는 `Error: <메시지>`를 stderr에 쓰고 종료 코드 1. 그 밖의 예외는 traceback과 함께 비정상 종료한다.
- 첫 인자가 `-`로 시작하면(`-h/--help` 제외) `run`이 앞에 붙은 것으로 처리한다.
- `--strategies` 허용값: `random`, `greedy`, `lcb`, `thompson`.

## 설정 YAML (`schema_version: 1`)

| 절 | 키 (기본값) | 제약 |
|---|---|---|
| 루트 | `schema_version` (1), `seed` (42), `mode` (`replay`), `output_dir` | seed ∈ [0, 2³¹−1), mode ∈ {replay, docking}, `output_dir`·`data.path` 필수 |
| `resources` | `cpu_threads` (8), `gpu_ids` ([0,1]), `device` (`cpu`), `deterministic` (true) | cpu 1–9, gpu ⊆ {0,1} 중복 없음, device ∈ {cpu, cuda:0, cuda:1}이고 GPU면 `gpu_ids`에 포함 |
| `data` | `path`, `id_column` (`Catalog ID`), `smiles_column` (`smiles`), `score_column` (`docking_score`), `invalid_policy` (`error`), `deduplicate` (true), `fingerprint_type` (`morgan`), `fingerprint_radius` (2), `fingerprint_bits` (2048) | policy ∈ {error, skip}, type ∈ {morgan, feature_morgan}, radius ≥ 0, bits ≥ 1 |
| `model` | `models` ([tabm]), `ensemble.strategy` (`mean`), `ensemble.weights`, `params`, TabM: `n_blocks` (3), `d_block` (128), `k` (16), `dropout` (0.1), `learning_rate` (0.001), `weight_decay` (0.0003), `epochs` (100), `batch_size` (128), `patience` (10), `validation_fraction` (0.2), `auto_tune` (false), `tuning_method` (`random`), `n_trials` (10), `inference_batch_size` (null) | 아래 |
| `learning` | `initial_size` (16), `batch_size` (8), `cycles` (5), `budget` (56), `strategy` (`lcb`), `beta` (1.0), `beta_schedule` (`constant`), `beta_final` (null) | 3 ≤ initial_size ≤ budget, cycles ≥ 0, beta ≥ 0 |
| `evaluation` | `holdout_fraction` (0.2), `split` (`random`), `top_k` (10), `save_predictions` (true) | fraction ∈ [0, 0.5), split ∈ {random, scaffold}, 도킹 모드면 fraction = 0 |
| `docking` | `backend` (`vina`), `receptor`, `reference_ligand`, `center`, `box_size` ([22,22,22]), `executable`, `exhaustiveness` (8), `n_poses` (3), `conformers` (1), `timeout` (300.0), `cpu` (1), `workers` (1) | 아래 |

`model` 제약:
- `models`: 비어 있지 않음, 대소문자 무관, 별칭(`tamb`→tabm, `rf`→random_forest, `et`→extra_trees) 정규화, 같은 백엔드 중복 금지. 루트 `models:` 단축형 허용, 단 `model.models`와 동시 사용 금지.
- `params.<이름>`: 선택한 모델만, `device/seed/random_state/n_jobs/cpu_threads` 금지. `params.tabm`은 평면 TabM 필드를 덮어쓴다.
- `ensemble.weights`: `weighted`에서만, 모델 수와 같은 길이, 유한·음이 아님·합 양수.
- `k ≥ 2`, `validation_fraction ∈ (0, 0.5)`, `dropout ∈ [0, 1)`, `learning_rate > 0`, `weight_decay ≥ 0`, `tuning_method ∈ {grid, random}`, `auto_tune`은 `tabm` 선택 시에만.
- `beta_schedule`: `constant`면 `beta_final` 금지, `linear`/`exponential`이면 `beta_final ≥ 0` 필수, `exponential`은 둘 다 양수.

`docking` 제약 (도킹 모드):
- `receptor` 필수(준비된 PDBQT, 원자 포함), `center`(숫자 3개) 또는 `reference_ligand`(PDB/PDBQT/SDF/MOL/MOL2, 중원자 무게중심; 다중 SDF는 첫 레코드) 중 하나 필수, `center`가 있으면 우선.
- `backend: qvina`면 `executable` 필수. `box_size`는 양수 3개.
- `cpu ≤ resources.cpu_threads`, `workers × cpu ≤ resources.cpu_threads`, `timeout > 0`.

foundation 백엔드 `params`(공통 `max_features` 128, `max_context_samples` 2048, `null`이면 해제):
- `nori`: `model` (`nori-6m`|`nori-30m`|`nori-100m`), `model_path`, `augmentations` ([] 또는 ["yj"]), `svd_dim`
- `tabldm`: `model_path`, `allow_auto_download`, `checkpoint_version`, `n_estimators` (8), `batch_size`, `kv_cache` (bool|`kv`|`repr`), `norm_methods`, `feat_shuffle_method` (`none`|`shift`|`random`|`latin`), `use_amp` (bool|`auto`), `offload_mode` (bool|`auto`|`cpu`|`disk`|`gpu`; `disk`면 `disk_offload_dir` 필수), `disk_offload_dir`
- `tabfm`: `checkpoint_path`, `n_estimators` (8), `batch_size`, `max_num_rows`, `norm_methods`
- `random_forest`/`extra_trees`: scikit-learn 생성자 인자(`n_estimators` 기본 128 등). `warm_start` 금지.

## 입력 CSV

- 파일 하나 또는 CSV 디렉터리(파일명 정렬 순서로 읽음). UTF-8 BOM, 따옴표 안의 쉼표·줄바꿈, 추가 메타데이터 열 허용(모델 입력에서 제외).
- 필수 열: ID 열, SMILES 열. replay 모드는 점수 열도 필수. 헤더는 고유해야 하고 모든 레코드의 필드 수가 헤더와 같아야 한다.
- `source_row`는 헤더를 1로 세는 논리 레코드 번호(내장 줄바꿈은 세지 않음).

## Python API

자원 설정은 수치 모듈 import 전에 한다.

```python
from active_learning_docking.config import load_config, ResourceConfig, ModelConfig
from active_learning_docking.resources import configure_resources
runtime = configure_resources(ResourceConfig(cpu_threads=2, device="cpu"), seed=42)
```

| 호출 | 입력 | 출력 | 오류 |
|---|---|---|---|
| `load_config(path, overrides=None)` | YAML 경로, `"a.b=v"` 목록 | 검증된 `ExperimentConfig` | `ValueError` |
| `run_experiment(config, runtime=None)` | 설정, 선택적 runtime | summary dict | `config.validate()` 실패, 출력 비어 있지 않음(`FileExistsError`), 자원 설정 실패는 매니페스트를 만들기 전에 발생. 그 뒤의 예외는 매니페스트에 `failed`와 오류를 기록한 뒤 재발생 |
| `configure_resources(resource_config, seed)` | `ResourceConfig`, 시드 | runtime dict(`device`=논리 장치 등) | `ValueError`(정책 위반), `RuntimeError`(CUDA 이미 초기화, CUDA 없음) |
| `regressors.create_regressor(model_config, seed=, device=, cpu_threads=)` | `ModelConfig` | 단일 백엔드 또는 `RegressionEnsemble` | `ValueError`, `ImportError`(런타임 없음) |
| `regressors.load_regressor(dir, device=)` | 저장된 모델 디렉터리 | 복원된 모델 | 해시·버전 불일치 `ValueError` |
| `model.TabMRegressor(...).fit(X, y, groups=None)` | X: bool/int/float 행렬 또는 `PackedFingerprints`, y: 유한 실수, groups: 행마다 비어 있지 않은 문자열 | self | 그룹 2개 미만, 비유한 값 `ValueError` |
| `.predict(X)` | X | `(mean, spread)` 1차원 배열, 점수 단위 | 미학습 `RuntimeError`, 특성 수 불일치 `ValueError` |
| `TabMRegressor.predict_members(X)` | X | `(n, k)` 멤버 예측 | 위와 같음 |
| `.save(dir)` / `TabMRegressor.load(dir, device=)` | 디렉터리 | 경로 / 복원 모델 | 가중치 해시 불일치 `ValueError` |
| `data.featurize(smiles, radius=2, n_bits=2048, fingerprint_type="morgan")` | SMILES 목록 | `uint8` 행렬 `(n, n_bits)` | 잘못된 SMILES `ValueError` |
| `store.PackedFingerprints.from_dense(X)`, `.take(indices)` | 밀집 행렬 / 인덱스 | 지연 압축 뷰 | — |

새 점수 출처: `pipeline.ActiveLearningExperiment`를 상속해 `make_oracle()`이 다음 객체를 반환하게 한다.
```python
evaluate(cid: str, smiles: str, output_dir: Path) -> {"score": float | None, "status": "ok" | "failed",
                                                       "error": str | None, "pose": str | None}
```
`status == "ok"`면 `score`는 유한해야 한다(아니면 실험 실패). `failed`면 `score`는 `None`이고 예산 1을 소모한다.

호환 진입점: `unified_model.UnifiedModel`(= `TabMRegressor`, `save_model`/`load_model` 철자 지원), `Activedocking.ActiveLearningWithConformers(config).run()`(`ExperimentConfig`만 허용).

## 실행 산출물

| 파일 | 내용 | 존재 조건 |
|---|---|---|
| `config.resolved.yaml` | 유효 설정, 절대 경로 | 항상 |
| `manifest.json` | `status`(running/completed/failed), 입력·코드 해시, 배포판 버전, Git revision/dirty, 자원, 완료 시 `artifact_sha256` | 항상 |
| `library.csv`, `rejected_rows.csv` | 수용 레코드 / 거부·중복 사유(`reason`, `kept_compound_id`) | CSV 입력일 때 |
| `split.csv` | `compound_id, partition`(pool/holdout) | 홀드아웃이 있을 때만 |
| `observations.csv` | 모든 시도: `compound_id, smiles, library_index, cycle, attempt, score, status, error, pose, prediction_before_query, uncertainty_before_query` | 항상 |
| `final_results.csv` | 성공 관측을 (점수, ID) 순 정렬 | 항상 |
| `predictions.csv` | `compound_id, smiles, prediction, uncertainty, partition, queried` | `save_predictions: true` |
| `metrics.csv` | 사이클별 `attempted, successful, failed, best_score, internal_validation_mse_standardized, acquisition_beta`, 앙상블 `ensemble_weight_*`/`ensemble_validation_mse_*`, 홀드아웃 `holdout_*`, replay `discovery_*` | 항상 |
| `training_history.csv` | 사이클·epoch별 손실(표준화 단위) | 항상 |
| `summary.json` | `mode, seed, strategy, models, pool_size, holdout_size, rejected_rows, budget, attempted, successful, cycles_completed, stop_reason, final_metrics` | 완료 시 |
| `active_learning.log` | 실행 로그 | 항상 |
| `model/` | 최종 모델 | 완료 시 |
| `cycle_NNN/queries.csv`, `training_split.csv`(train/validation/unused), `metrics.json`, `model/` | 사이클 기록 | 사이클마다 |
| `cycle_NNN/pool_predictions_before_acquisition.csv` | 선택 직전 풀 예측 | 사이클 ≥ 1이고 `save_predictions: true` |
| `cycle_NNN/holdout_predictions.csv` | 홀드아웃 예측 | 홀드아웃이 있을 때 |
| `cycle_NNN/member_training_splits.csv` | 멤버별 분할(`model` 열 포함) | 앙상블 |
| `cycle_NNN/ensemble_validation_split.csv` | weight_train/weight_validation | `strategy: validation` |
| `cycle_NNN/poses/<slug>_<hash16>/` | `request.json, result.json, worker.log, conformer_NNN.pdbqt, poses_NNN.pdbqt, best_poses.pdbqt` (QVina는 `engine_NNN.log`) | 도킹 모드 |

`stop_reason` ∈ {`budget`, `pool_exhausted`, `cycles`}.

## 모델 체크포인트

| 모델 | 파일 | 복원 |
|---|---|---|
| TabM | `tabm_weights.pt` + `model_metadata.json` | 해시 확인 후 `weights_only=True` 로드 |
| RF/ET | `training_data.npz` + 메타데이터 | 같은 scikit-learn 버전에서 결정적 재학습 |
| foundation | `foundation_context.npz` + 메타데이터(패키지 버전, HF 저장소/revision, 체크포인트 경로·해시, 문맥·특성 인덱스) | 런타임 버전·해시 확인 후 재생성 |
| 앙상블 | 멤버별 하위 디렉터리(`model/tabm/` 등) + 앙상블 메타데이터 | 멤버 복원 + 저장된 가중치 |

모든 메타데이터의 `model_info.model_type`이 로더 분기 키다. 복원은 예측 재현용이며, 옵티마이저 상태나 중단된 실험 재개를 포함하지 않는다.

## 라이브러리 저장소 디렉터리

`compound_id.npy`, `smiles.npy`, `source.npy`, (replay) `scores.npy`, 비트 압축 `fingerprints.npy`, `rejected_rows.csv`, `library.json`(설정, 입력 해시, 파일별 체크섬). ID 정렬. 열 때 모든 체크섬을 검증한다. `data.path`에 이 디렉터리를 주면 저장소 입력으로 인식된다.
