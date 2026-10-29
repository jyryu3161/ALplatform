# active_learning_docking 패키지

## 범위

설치 가능한 패키지 전체: 설정 검증, 자원 제한, 라이브러리 로드·저장소, 회귀 백엔드와 앙상블, 질의 선택, 오라클(replay·도킹), 지표, 실험 단계, 산출물 기록, 재현 비교, CLI.

범위가 아닌 것:
- 수용체 준비, replay fixture 생성, 설치 검증 → `scripts/`
- 설치·잠금 → `install.sh`, `requirements/`
- 버전 1 호환 진입점 → 루트의 `main.py`, `Activedocking.py`, `unified_model.py`, `__init__.py` (제거 예정, 새 기능 추가 금지)
- 번들 fixture → `examples/`, `inputs/` (이 패키지 코드가 실행 중에 고치지 않는다)

## 모듈별 소유와 불변 조건

**`config.py`**
- 최상위 import는 표준 라이브러리, `yaml`, `.model_registry`뿐이다. numpy/torch/RDKit을 넣지 않는다.
- `_construct`가 알 수 없는 키·타입 불일치를 거부하고, `_UniqueLoader`가 중복 키를 거부한다. 새 필드는 데이터클래스 + `validate()` 조건 + (경로형이면) `from_dict`의 경로 해석 목록에 함께 추가한다.
- `float` 필드는 int도 받지만 비유한 값은 거부한다. `bool`은 `int`로 통과시키지 않는다(`type(value) is not annotation`).
- 자원 상한(CPU 1–9, GPU ⊆ {0,1})은 여기와 `resources.py`, `docking.py`(cpu), `store.py`(workers), `model.py`·`foundation.py`(device 집합, `CUDA_VISIBLE_DEVICES` 토큰), `scripts/`의 두 준비 스크립트(`--cpu`)에 각각 있다. 하나만 바꾸지 않는다.
- `ModelConfig.models` 기본값은 `["tabpfn"]`이다. `tabm`이 없으면 `TABM_ONLY_FIELDS`(`n_blocks, d_block, k, dropout, learning_rate, weight_decay, epochs, patience, tuning_method, n_trials`)가 데이터클래스 기본값과 다를 때 `ValueError`. `batch_size`, `validation_fraction`, `inference_batch_size`는 공용이라 넣지 않는다. TabM 전용 필드를 추가하면 이 튜플에도 넣는다.
- `validate()`: `tabpfn` ∈ `models` + `device == "cpu"` + `budget > 1000` → 오류(앙상블 멤버 포함, 1000은 통과).
- `docking.rescoring` ∈ {`none`, `gnina`}. `gnina`는 `mode: docking` + `gnina_image` + `apptainer_executable` 필수, `none`이면 두 값이 `None`이어야 한다. 두 키는 `from_dict` 경로 해석 목록에 있다. 꺼져 있을 때 세 키는 `config.resolved.yaml`/매니페스트 `config`에만 나타나고 다른 산출물은 바뀌지 않아야 한다.

**`resources.py`**
- `configure_resources`는 numpy/torch를 함수 안에서 import한다. 호출 전에 다른 모듈이 torch를 import해 CUDA를 초기화했다면 `RuntimeError`를 내야 한다.
- 반환 dict의 `threadpool_controller`는 프로세스 수명 동안 살아 있어야 한다(제한 유지). 매니페스트에는 이 키를 빼고 기록한다.
- affinity는 허용 집합을 정렬해 앞의 `cpu_threads`개. 물리 GPU → 논리 `cuda:N` 변환은 허용 목록 안 위치.

**`data.py`**
- 거부 사유 문자열(`missing_compound_id`, `missing_smiles`, `invalid_smiles`, `invalid_score`, `nonfinite_score`, `duplicate_compound_id`, `duplicate_structure`, `conflicting_compound_id`, `conflicting_structure_score`)은 `rejected_rows.csv` 계약이다. 이름을 바꾸지 않는다.
- 대표 ID는 같은 구조 중 사전순 최소 ID, 출력은 ID 정렬. `store.py`의 구축 경로와 같은 결과를 내야 한다 — 한쪽 규칙을 바꾸면 다른 쪽도 바꾼다.
- `featurize`는 `uint8`, 잘못된 구조는 0벡터가 아니라 오류. 점수·ID를 특성에 넣는 경로를 만들지 않는다.
- `split_holdout`은 라벨을 받지 않는다. 그룹을 통째로 옮기고 최소 한 그룹을 풀에 남긴다.

**`store.py`**
- 저장소는 ID 정렬 열 + 비트 압축 지문 + `library.json` 체크섬. `open_library_store(verify=True)`는 모든 체크섬을 검증한다.
- `PackedFingerprints.take`는 지연 뷰, `__getitem__`/`dense`만 실제로 푼다. 파이프라인 경로에서 전체 풀을 `dense()`로 만들지 않는다(10^8에서 메모리 폭발).
- 구축 실패 시 부분 출력을 지운다. 구축 결과는 워커 수와 무관하게 바이트 동일해야 한다.

**`model.py` (`TabMRegressor`)**
- `tabm.TabM.make`만 사용. 구조 복사·재정의, MC dropout, 자체 MLP 금지.
- 내부 검증은 `groups`(정규 SMILES) 단위, 목표 표준화는 내부 학습 행만, 상수 목표면 scale 1. 최저 검증 손실 체크포인트를 복원한다.
- `fit/predict/predict_members`는 bool/int/float 행렬과 `PackedFingerprints`를 받고 `inference_batch_size` 배치마다 float32로 변환한다. uint8 입력과 float32 입력 결과가 같아야 한다.
- 저장: `tabm_weights.pt` + `model_metadata.json`(가중치 SHA-256), 로드: 해시 확인 후 `weights_only=True`.
- `_features`, `_targets`는 `regressors.py`도 쓰는 공용 검증 함수다.

**`regressors.py`**
- `create_regressor`는 선택된 모델만 생성한다(하나면 단일 백엔드, 여럿이면 `RegressionEnsemble`). `tabm`·트리가 아닌 이름은 모두 `FoundationRegressor`로 간다(`tabpfn` 포함). 앙상블 `validation` 전략의 그룹 수 규칙에서 트리가 아닌 멤버(TabPFN 포함)가 있으면 3개 이상.
- 앙상블 멤버 시드 = `(seed + index) % 2**32`. 가중치는 멤버 평균에, 결합 분산은 `Σ w(σ² + (μ − mean)²)`.
- `validation` 전략: 그룹 분할 → 멤버별 탐색 학습·MSE(하한 1e-12) → 멤버를 새로 만들어 전체 재학습. 탐색 모델은 재학습 전에 해제한다(대형 문맥·GPU 메모리).
- `ForestRegressor`는 `random_state`, `n_jobs`, `warm_start`를 거부하고, 복원은 같은 scikit-learn 버전에서 재학습.
- 체크포인트 로더 분기 키는 `model_metadata.json`의 `model_info.model_type`.

**`foundation.py`**
- 외부 호스팅 API를 쓰지 않는다. Hugging Face 다운로드는 항상 `token=False`와 고정 커밋 revision(Nori는 `_NORI_CHECKPOINTS`, TabLDM·TabFM은 체크포인트 해석 함수 안)으로 한다. 브랜치명이나 `main`으로 바꾸지 않는다.
- 복원 시 설치된 upstream 소스 revision이 저장 당시와 다르면 실패한다.
- 문맥 = 획득 관측에서 시드 비복원 추출(`max_context_samples`), 특성 = 그 문맥의 분산 상위 `max_features`(동점은 원래 인덱스 순). 질의 행은 선택에 쓰지 않는다. **TabPFN만 순서가 다르다**: 특성은 획득 관측 전체(`X[:]`)의 분산(이진 지문은 정수 키 `c·(n−c)`로 동점을 정확히 판정) 상위 `max_features`, 그 다음 문맥 추출. 이 분기를 nori·tabldm·tabfm에 적용하지 않는다(기존 결과가 바뀐다).
- 문맥 행 2개 미만, 변하는 비트 없음은 모델 할당 전에 실패. TabPFN은 고른 비트가 문맥 안에서 하나도 변하지 않아도 실패.
- 지원 파라미터 목록(`_COMMON`, `_SUPPORTED`) 밖의 키는 거부. TabFM 캐시 관련 옵션은 거부. TabPFN은 `max_features` 1–500, `max_context_samples` 2–10000 정수(`null` 불가, `_TABPFN_LIMITS`), 기본 500/10000. `model_path: null`은 고정 다운로드 기본값.
- `preflight()`는 런타임·가중치를 확인해 도킹 예산을 쓰기 전에 실패시킨다. TabPFN은 `tabpfn.model_loading.download_model`과 `tabpfn.settings.settings.tabpfn`이 있는지도 확인한다(실행 중 이 둘을 고정·차단하므로).
- TabFM spread는 0이다(공개 API가 점 예측만 제공). TabPFN spread는 `output_type="full"`의 bar-distribution 분산으로 계산한 정확한 표준편차(`_distribution_std`, 점수 단위, 상수 목표면 0)다. 분위수 근사로 바꾸지 않는다.
- TabPFN 체크포인트는 `_TABPFN_CHECKPOINT`(저장소 `Prior-Labs/TabPFN-v2-reg`, 고정 revision, 파일명 `tabpfn-v2-regressor.ckpt`, SHA-256)만 허용한다(`_check_tabpfn_reference`, 로컬 `model_path`도 같은 파일명·해시). `tabpfn`은 파일명으로 모델 판을 추론하므로 이름을 바꾸지 않는다. 2.5 이상 체크포인트를 쓰지 않는다.
- TabPFN 실행 옵션은 `_TABPFN_OPTIONS`(사용자 설정 아님), `tabpfn.settings` 값은 `_TABPFN_SETTINGS`로 실행 중 고정하고(`_pinned_upstream_state`, `.env`/`TABPFN_*` 영향 차단) 끝나면 되돌린다. 같은 범위에서 `download_model`을 거부 함수로 바꿔 라이브러리 자체 다운로드를 막는다. 둘 다 `model_info.fixed_upstream_options`에 기록되고, 로드 시 현재 상수와 다르면 실패한다. 옵션을 바꾸면 이전 체크포인트와 결과가 달라진다.
- 외부 통신은 고정 체크포인트 `hf_hub_download(..., token=False)`뿐이다. `tabpfn-client`(호스팅 API)를 쓰지 않는다. `tabpfn`을 올릴 때 telemetry·인증 경로가 생겼는지 확인한다.

**`acquisition.py`**
- 모든 전략은 최소화. `random`/`thompson`은 명시적 `np.random.Generator`가 필요하다.
- 동점은 위치(= ID 순) 또는 주어진 ID. 큰 풀에서는 `argpartition` 후 경계 동점을 명시적으로 해결한다 — 이 처리를 단순화하면 사전순으로 우선인 화합물이 빠질 수 있다.
- β는 유한·음이 아님. `scheduled_beta`의 사이클 번호는 1부터.

**`evaluation.py`**
- 정의되지 않거나 넘치는 지표는 `None`. 발견 지표에는 풀만 넘긴다(홀드아웃 포함 금지).

**`docking.py`**
- 화합물 디렉터리 = `<slug>_<SHA-256(JSON[cid, smiles]) 앞 16자>`. 비어 있지 않거나 링크면 작업 프로세스를 띄우기 전에 `FileExistsError`(실험 중단).
- 작업 프로세스는 `python <이 파일 절대경로> --worker request result`, 새 세션, 스레드 1, GPU 없음. 시간 초과·중단 → 프로세스 그룹 SIGKILL 후 `_kill_session`으로 같은 세션 ID의 남은 프로세스(Apptainer `squashfuse_ll` 등) SIGKILL, `failed` 기록. 이 정리는 오류를 무시하고 원래 예외를 바꾸지 않는다.
- `TimeoutExpired`와 일반 예외는 `failed` 레코드로 바뀌고 예산을 소모한다. 성공 점수는 유한해야 하고 포즈 파일이 있어야 한다.
- 리간드: 분리된 SMILES 거부, ETKDGv3(시드, 스레드 1) → MMFF94(없으면 UFF) → Meeko. 형태별 도킹 후 최소 에너지(동점은 낮은 형태 번호)를 `best_poses.pdbqt`로 복사.
- 덮어쓰기 방지: 작업 프로세스는 시작 시 모든 형태·포즈·엔진 로그·최적 포즈 경로가 없는지 먼저 확인하고, `request.json`·형태 PDBQT·`best_poses.pdbqt`·작업 프로세스의 `result.json`은 `"x"`/`"xb"` 모드로 만든다. Vina 포즈는 `overwrite=False`로 쓴다. 예외: `worker.log`, QVina `engine_NNN.log`는 `write_text`이고, 부모 프로세스는 작업 프로세스가 쓴 `result.json`을 최종 레코드(시드·실패 사유 포함)로 다시 쓴다.
- 엔진 대체 금지: `vina`면 Python 패키지, `qvina`면 지정 실행 파일. 없으면 실패.
- GNINA 재채점(`rescoring: gnina`)
  - 오라클 생성자가 `_gnina_provenance`로 Apptainer·SIF 존재, `apptainer --version`, 이미지 안 `gnina --version`에서 `gnina v` 줄이 정확히 하나이고 `GNINA_VERSION_PREFIX`(`gnina v1.3.3 `)로 시작하는지, SIF SHA-256을 확인해 `self.rescoring`(매니페스트 기록)에 둔다. 예산 사용 전이다. PATH를 찾지 않는다. 꺼져 있으면 `request.json`에 재채점 키를 넣지 않는다(정확 일치 게이트).
  - 작업 프로세스 안에서 도킹 직후 실행한다. 대상은 최고 형태 포즈 파일에서 기록 에너지와 (출력 자릿수 반올림으로) 같은 **첫 MODEL 하나**(`_select_model`) → `rescore_pose.pdbqt` → Meeko·RDKit로 SMILES 결합 차수를 입힌 단일 레코드 `rescore_pose.sdf`(`_pose_to_sdf`, 정규 SMILES 불일치면 실패) → GNINA. `best_poses.pdbqt`는 지금처럼 전체 포즈 파일 복사.
  - 명령: `_apptainer_exec`(`--containall --cleanenv --no-home --net --network none --pwd / --env CUDA_VISIBLE_DEVICES=`, 바인드는 수용체 `:ro`와 화합물 디렉터리만, `--nv` 금지) + `gnina --score_only --cnn_scoring rescore --cnn <GNINA_CNN_MODELS 순서대로> --no_gpu --cpu <docking.cpu> --seed s(화합물 시드, "gnina", SMILES)`. 환경에서 `APPTAINER_*`/`SINGULARITY_*`를 지운다. GNINA는 `subprocess.run`으로 작업 프로세스와 **같은 세션**에서 띄운다(시간 초과 정리 대상).
  - 값은 `gnina_scored.sdf`의 `CNNscore`·`CNNaffinity` 데이터 항목에서 읽는다(`_gnina_values`: 레코드 정확히 1개, 항목 각 1개, 유한). 표준 출력 값을 쓰지 않는다. 성공 결과는 `score = −CNNaffinity`, `pose = rescore_pose.sdf`, `docking_energy`·`cnn_score`·`cnn_affinity`. 부모는 세 값이 비유한이면 실패로 바꾼다.
  - `RESCORING_FILES` 네 개는 작업 프로세스 시작 시 존재 점검 대상이다. `rescore_pose.pdbqt`·`rescore_pose.sdf`·`gnina.log`는 `"x"` 모드, `gnina_scored.sdf`는 실행 직전 부재 확인.
  - 실패(변환 실패, GNINA 비정상 종료, 항목 없음, 비유한, 시간 초과)는 화합물 실패이며 QVina/Vina 에너지로 점수를 채우지 않는다. `output_dir`의 `:`·`,`는 설정 검증(`ExperimentConfig.validate`)에서, 수용체 경로의 `:`·`,`는 생성자에서 거부한다(모두 예산 사용 전). 작업 프로세스 안의 화합물 디렉터리 검사는 마지막 방어선으로 남긴다.

**`pipeline.py`**
- 단계 순서: preflight → `prepare_library` → (`select_queries` → `query_oracle` → `fit_model` → `record_metrics`) × 사이클 → `finalize`.
- 사이클 시작 전 `allowance = budget − 시도 수`, `remaining()`이 0이거나 allowance ≤ 0이면 멈춘다.
- `fit_model`은 직전 모델을 `None`으로 놓은 뒤 새로 만든다(GPU 메모리). 시드 = `config.seed + cycle`.
- 초기 표본과 선택용 난수는 `SeedSequence(seed).spawn(2)`의 두 생성기로 분리되어 있다. 이 순서를 바꾸면 모든 기존 실행과 결과가 달라진다.
- 예측을 쓰는 부분에서 `self.features.take(...)`로 지연 뷰를 넘긴다.
- `run_experiment`는 실패 시 매니페스트에 `failed`를 쓰고 `LibraryValidationError.rejections`를 `rejected_rows.csv`로 남긴 뒤 예외를 다시 던진다.
- `run`은 `make_oracle()` 직후 `oracle.rescoring`이 있으면 매니페스트에 `rescoring` 키를 넣고 즉시 다시 쓴다(첫 질의 전).
- `docking.rescoring == "gnina"`일 때만 관측 레코드에 `docking_energy, cnn_score, cnn_affinity`를 기존 열 뒤에 붙인다(실패 행은 셋 다 `None`). 꺼져 있을 때 열을 추가하면 정확 일치 게이트가 깨진다.

**`provenance.py` / `experiments.py`**
- `artifact_hashes`는 `manifest.json`과 `active_learning.log`만 제외하고 모든 파일을 해시한다. 실행 디렉터리에 새 산출물을 추가하면 자동으로 무결성·비교 대상이 된다.
- `compare_runs`는 수치 허용 오차 없이 비교하고, `pose` 열의 절대 경로만 실행 디렉터리 기준 상대 경로로 바꿔 비교한다. `split.csv`, `predictions.csv`는 양쪽 모두 없을 때만 건너뛴다. 포즈 바이트 비교(`all_pose_bytes`)는 매니페스트 해시 목록에서 `.pdbqt`와 `.sdf`(GNINA 입력·출력)를 모두 대상으로 한다. 새 포즈 형식을 추가하면 여기에도 넣는다.
- `reproduce`/`benchmark`는 설정을 YAML로 써서 새 인터프리터에서 `run`한다. 같은 프로세스 안에서 반복 실행하도록 바꾸지 않는다.

**`cli.py`**
- `run`, `prepare-library`는 `configure_resources` 다음에 `pipeline`/`store`를 import한다. `prepare-library`는 GPU를 쓰지 않도록 `device` 기본값(cpu)의 `ResourceConfig`로 자원을 설정한다.
- 처리하는 예외 집합(`ValueError, OSError, RuntimeError, ImportError, yaml.YAMLError`) → `Error:` + 종료 1.

## 테스트 지침

- 테스트는 `tests/`에 모듈별 파일(`test_<모듈>.py`)로 둔다. `conftest.py`가 CPU 2개·스레드 1로 고정한다.
- 반드시 테스트할 것:
  - 홀드아웃 라벨을 바꿔도 질의 순서·학습 기록·예측이 바뀌지 않음
  - 실패한 시도의 예산 소모와 점수 미기입
  - 같은 분자 그룹이 분할 경계를 넘지 않음
  - 행·파일 순서를 바꿔도 대표 ID·선택이 같음
  - CSV 입력과 저장소 입력의 결과 정확 일치
  - 체크포인트 저장→로드 후 예측 정확 일치
  - 도킹 워커 1개와 2개의 결과 일치(통합 테스트)
- 경계 사례: 상수 목표값, 분자 그룹 1개, 성공 관측 2개, 풀 3개 미만, 빈 질의 수, β = 0, `cycles = 1`의 스케줄, 넘치는 regret, 비어 있지 않은 출력 디렉터리, 끊긴 심볼릭 링크, 비숫자 `CUDA_VISIBLE_DEVICES`.
- 무거운 선택 런타임은 가짜 upstream 클래스로 계약 테스트를 하고, 실제 가중치 테스트는 `ALD_RUN_FOUNDATION_TESTS=1` + `-m foundation`, 실제 도킹은 `ALD_RUN_DOCKING_TESTS=1` + `-m integration`으로 opt-in. GPU 테스트는 `gpu` 마커.
- TabPFN: 기본 테스트는 가짜 upstream으로 파라미터 범위·`null` 거부, 전체 관측 기준 특성 선택과 동점, 문맥 추출, 저장→로드 일치, 해시·revision 불일치 거부, spread = 분포 표준편차를 확인한다. 실제 가중치는 `test_real_public_foundation_inference_and_restore[tabpfn]`과 `test_real_tabpfn_replay_runs_are_exact_and_holdout_blind`(이름을 바꾸면 `scripts/verify_install.py`의 `FOUNDATION_TESTS`도 고친다).
- GNINA: 기본 테스트는 테스트가 만드는 가짜 apptainer 실행 파일(설정 경로로 지정)로 명령·격리 옵션·실패 규칙·버전 불일치·예산 소모를 확인한다. 실제 GNINA는 `ALD_RUN_DOCKING_TESTS=1`과 `ALD_GNINA_IMAGE`(GNINA 1.3.3 SIF), `ALD_APPTAINER`(실행 파일)가 모두 있어야 돈다(`tests/test_pipeline.py`, 반복 실행·워커 1/2 정확 일치). 없으면 skip이고, skip을 통과로 세지 않는다.
- 테스트에서 기본 모델에 기대지 말고 필요한 모델을 명시한다. 기본 모델이 TabPFN이라 `models`를 생략한 설정은 TabPFN 런타임이 필요하다.
