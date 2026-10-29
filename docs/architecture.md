# 시스템 구성

## 구성 요소와 연결

설치 가능한 Python 패키지 `active_learning_docking` 하나가 시스템의 전부이며, 서버나 데이터베이스 없이 한 프로세스(와 그 자식 프로세스)로 실행된다. 입력은 YAML 설정 파일과 화합물 CSV(또는 미리 만든 라이브러리 저장소), 출력은 새로 만든 실행 디렉터리 하나다.

```text
YAML + --set ──► config ──► resources ──► pipeline.ActiveLearningExperiment
                (스키마 검증)  (CPU/GPU 제한,   │
                               시드 고정)       ├─ data / store   : 라이브러리 로드·지문·홀드아웃
                                                ├─ regressors     : 모델 생성 (tabpfn 기본 / tabm / 트리 / foundation / 앙상블)
                                                ├─ acquisition    : 다음 질의 선택
                                                ├─ oracle         : ReplayOracle 또는 docking.DockingOracle(선택적 GNINA 재채점)
                                                ├─ evaluation     : 회귀·발견 지표
                                                └─ provenance     : 매니페스트·해시·JSON 기록
experiments (reproduce / benchmark) ──► 새 Python 프로세스에서 `run`을 반복 실행 후 결과 비교
```

**import 순서가 구조의 일부다.** `cli.py`는 `config.py`만 최상위에서 import하고, `run`·`prepare-library` 명령에서 `resources.configure_resources`를 먼저 호출한 뒤에야 `pipeline`·`store`를 import한다. 이 순서 덕분에 BLAS/OpenMP/PyTorch 스레드 풀이 이미 제한된 CPU affinity와 `CUDA_VISIBLE_DEVICES`를 물려받는다. `pipeline.py`를 import하면 최상위의 `store` → `data` 사슬로 numpy·pandas·RDKit이 곧바로 로드되고, PyTorch·모델·도킹 모듈은 각 단계 함수 안에서 로드된다. 따라서 `run_experiment(config)`를 직접 호출하는 경로는 그 안에서 자원 정책을 적용하더라도 numpy·RDKit이 정책보다 먼저 로드된다. 엄격한 제한이 필요하면 `configure_resources`를 호출한 뒤 `pipeline`을 import한다.

## 모듈 지도

| 모듈 | 역할 | 의존 방향 |
|---|---|---|
| `config.py` | 버전 1 데이터클래스 스키마, 엄격한 YAML 로드, `--set` 덮어쓰기, YAML 디렉터리 기준 경로 해석, 기본 모델(`[tabpfn]`)·TabM 전용 값 검사·TabPFN CPU 예산 규칙·재채점 키 검사 | `model_registry`만 의존. 수치 라이브러리 import 없음 |
| `model_registry.py` | 백엔드 정식 이름·별칭·설치 안내 | 의존 없음 |
| `resources.py` | CPU affinity, 스레드 수, 물리 GPU 마스킹, 난수 시드, 결정론적 연산 설정 | numpy·torch·threadpoolctl을 함수 안에서 import |
| `data.py` | CSV 검증, 정규 SMILES, 중복 제거, Morgan/feature-Morgan 지문, 그룹 홀드아웃 분할 | RDKit |
| `store.py` | `PackedFingerprints`, `Library`, 병렬 스트리밍 `build_library_store` | `data`의 검증 규칙 재사용 |
| `model.py` | `TabMRegressor` (공식 `tabm==0.0.3` 학습·튜닝·체크포인트) | torch, tabm, `store` |
| `regressors.py` | 모델 팩토리, 랜덤 포레스트/엑스트라 트리, `RegressionEnsemble`, 통합 로더 | `model`, `foundation`, `store`, `config` |
| `foundation.py` | TabPFN v2(기본 모델)·Nori·TabLDM·Google TabFM 로컬 어댑터, 문맥 행/특성 선택(TabPFN은 획득 전체 기준 특성 선택 후 문맥 추출), 고정 체크포인트·해시, TabPFN 설정 고정·다운로드 차단 | `tabpfn`(핵심 의존성), 선택 설치 패키지 |
| `acquisition.py` | `random`/`greedy`/`lcb`/`thompson` 선택과 beta 스케줄 | numpy |
| `evaluation.py` | 회귀 지표와 발견 지표 | numpy, scipy |
| `docking.py` | 리간드 준비, 격리된 Vina/QVina 작업 프로세스, `DockingOracle`, 선택적 GNINA 재채점(포즈 MODEL 선택 → SDF 변환 → Apptainer로 GNINA 실행 → 값 읽기), GNINA 버전·SIF 해시 확인, 시간 초과 시 세션 전체 정리 | RDKit, Meeko, vina(선택), 외부 Apptainer·GNINA SIF(선택) |
| `pipeline.py` | `ActiveLearningExperiment` 단계, `ReplayOracle`, `run_experiment` | 위 모듈 전부 |
| `provenance.py` | SHA-256, 환경 매니페스트, 원자적 JSON, 출력 경로 사전 점검 | 표준 라이브러리 |
| `experiments.py` | `reproduce`(새 프로세스 2회 실행 후 정확 비교, 포즈 바이트는 `.pdbqt`·`.sdf`), `benchmark`(시드 쌍 비교) | 하위 프로세스로 `run` 호출 |
| `cli.py` | `run`, `validate`, `models`, `reproduce`, `benchmark`, `prepare-library` | `config` → (자원 설정 후) 나머지 |

패키지 밖 구성:
- `scripts/prepare_receptor.py`, `scripts/build_replay.py` — 수용체 준비와 replay 점수 fixture 생성. 패키지의 `provenance.validate_new_paths`와 `docking` 모듈을 사용한다.
- `scripts/verify_install.py` — `install.sh`가 호출하는 설치 검증기.
- `install.sh` + `requirements/install-{cpu,cu128}.lock` — uv 기반 고정 설치.
- `main.py`, `Activedocking.py`, `unified_model.py`, 루트 `__init__.py` — 버전 1 사용자를 위한 호환 진입점. 새 기능은 이곳에 넣지 않는다.

## 대표 흐름: `active-learning-docking run --config X.yaml`

1. `config.load_config` — 중복 키를 거부하는 YAML 로더로 읽고, `--set section.key=value`를 YAML 스칼라로 덮어쓴 뒤 데이터클래스로 엄격하게 구성하고 `validate()`. 상대 경로(`data.path`, `output_dir`, `docking.receptor/reference_ligand/executable/gnina_image/apptainer_executable`, 백엔드 `model_path/checkpoint_path/disk_offload_dir`)는 YAML 파일 디렉터리 기준 절대 경로가 된다.
2. `resources.configure_resources` — affinity를 처음 `cpu_threads`개 CPU로 줄이고, 스레드 환경 변수를 고정하고, 물리 GPU ID를 논리 `cuda:N`으로 바꾼다.
3. `run_experiment` — 출력 디렉터리가 비어 있지 않으면 즉시 실패. `config.resolved.yaml`과 `manifest.json`(status 미완료)을 먼저 쓴다.
4. `ActiveLearningExperiment.run`
   - **preflight**: 선택한 모델을 한 번 생성하고 foundation 런타임·가중치를 확인한다(기본 TabPFN이면 고정 체크포인트를 받거나 찾아 해시를 확인). 도킹 예산을 쓰기 전에 실패시키기 위함이다.
   - **prepare_library**: CSV면 검증·지문 생성 후 `library.csv`/`rejected_rows.csv` 기록, 저장소면 memory-map으로 열고 설정 일치 확인. 그 다음 라벨을 보지 않고 홀드아웃을 봉인한다(`split.csv`는 홀드아웃이 있을 때만).
   - **make_oracle**: 오라클을 만든다. `rescoring: gnina`면 `DockingOracle` 생성자가 Apptainer·SIF 존재, `apptainer --version`, 이미지 안 `gnina --version`(1.3.3 확인), SIF SHA-256을 계산하고, 파이프라인이 그 기록을 매니페스트 `rescoring` 키로 즉시 다시 쓴다. 첫 질의 전이다.
   - **사이클 0**: 남은 풀에서 `initial_size`개를 무작위 선택 → 오라클 평가 → 모델 학습 → 지표.
   - **사이클 1..cycles**: 풀 전체 예측(`inference_batch_size` 단위) → beta 계산 → 선택 → 오라클 평가(도킹이면 `docking.workers`개 스레드가 각자 자식 프로세스 실행) → 성공 관측 전체로 새 모델 학습 → 지표.
   - 예산 소진, 풀 소진, 사이클 상한 중 먼저 오는 조건에서 멈춘다.
   - **finalize**: 최종 모델, `final_results.csv`, 선택적으로 `predictions.csv`, `training_history.csv`, `summary.json`.
5. 성공 시 매니페스트에 `status: completed`와 모든 산출물의 SHA-256을 기록한다. 실패 시 `status: failed`와 오류를 기록하고 예외를 다시 던진다.

## 경계

- **오라클 경계**: 파이프라인과 점수 출처는 `evaluate(cid, smiles, output_dir) → {score, status, error, pose}` 하나로만 연결된다. 넘어가는 것은 화합물 ID·SMILES·출력 디렉터리이고, 모델 예측이나 다른 화합물 정보는 넘어가지 않는다.
- **도킹 프로세스 경계**: `DockingOracle`은 화합물마다 `request.json`을 쓰고 `python <docking.py 절대경로> --worker request result`를 새 세션(새 프로세스 그룹)으로 실행한다. 자식에는 스레드 1개 변수와 빈 `CUDA_VISIBLE_DEVICES`가 넘어가고, 결과는 `result.json`으로만 돌아온다. 시간 초과·중단 시 프로세스 그룹을 SIGKILL한 뒤 같은 세션 ID의 남은 프로세스도 모두 SIGKILL한다(`_kill_session`, `/proc` 검사).
- **GNINA 경계**(`rescoring: gnina`): 같은 작업 프로세스가 도킹 직후 `apptainer exec ... <SIF> gnina ...`를 자식 프로세스로 실행한다(데몬 없음, 같은 세션). 컨테이너에는 수용체(읽기 전용, `/input/receptor.pdbqt`)와 화합물 디렉터리(`/work`)만 바인드되고, 네트워크·GPU·호스트 환경·홈은 없다. 입력은 `rescore_pose.sdf`, 출력은 `gnina_scored.sdf`·`gnina.log`이며 값은 출력 SDF의 데이터 항목으로만 돌아온다.
- **라이브러리 경계**: 모든 단계가 화합물에 위치(ID 정렬 순서)로 접근한다. `Library`(문자열 열)와 `PackedFingerprints`(비트 단위로 압축된 행, 배치마다 풀어냄)가 CSV 입력과 10^8개 저장소를 같은 코드로 다루게 한다.
- **reproduce 경계**: `experiments._run`은 설정을 YAML로 써서 `python -m active_learning_docking run`을 새 인터프리터로 실행한다. 같은 프로세스 안의 상태가 비교에 섞이지 않는다.

## 외부 시스템 의존

| 대상 | 사용 시점 | 형태 |
|---|---|---|
| Hugging Face Hub | foundation 모델 첫 실행(기본 TabPFN 포함, 또는 `--test-models`) | 고정 revision 다운로드(`token=False`) 후 파일 해시 기록. TabPFN은 고정 해시와 다르면 실패 |
| Apptainer + GNINA 1.3.3 SIF | `rescoring: gnina` 도킹 모드 | 사용자가 미리 준비한 로컬 파일(`docking.apptainer_executable`, `docking.gnina_image`). 실행 중 네트워크 없음. 원본은 공식 Docker Hub 이미지 다이제스트 |
| GitHub (astral-sh/uv 릴리스, xiaomi-tabldm 고정 커밋) | `install.sh` | uv 압축 파일 SHA-256 검증, TabLDM은 Git 커밋 고정 |
| PyPI / PyTorch 인덱스 | `install.sh` | 잠금 파일의 정확한 버전 |
| AutoDock Vina Python 패키지 / QVina 실행 파일 | 도킹 모드 | 작업 프로세스 안에서 import / `docking.executable` 경로를 하위 프로세스로 실행. 저장소에 `bin/qvina2.1`이 동봉되어 QVina 통합 테스트가 사용 |
| `nvidia-smi` | GPU 실행 시 매니페스트 기록 | 실패해도 `uuid_driver: null`로 계속 |

실행 중 네트워크가 필요한 것은 캐시되지 않은 foundation 가중치 다운로드뿐이다. 기본 모델이 TabPFN이므로 새 환경의 첫 실행(CI의 replay `reproduce` 포함)은 약 44 MB의 TabPFN v2 체크포인트를 받는다. GNINA 재채점은 네트워크를 쓰지 않는다.
