# 프로젝트 규칙

## 검증 게이트

- **커밋 전 필수**: (1) 자원 제한 안에서 전체 테스트 통과, (2) `ruff check` 통과. 둘 중 하나라도 실패하면 커밋하지 않는다.
  ```bash
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=0,1 taskset -c 0,1 .venv-ald/bin/python -m pytest -q
  .venv-ald/bin/ruff check active_learning_docking tests scripts main.py unified_model.py Activedocking.py
  ```
- **구조 변경(리팩터링, 성능 개선, 내부 표현 변경) 시 추가 필수**: 변경 전후로 번들 설정을 새 디렉터리에 실행하고 `experiments.compare_runs(변경전, 변경후)`가 `passed: true`여야 한다. 수치 허용 오차 없이 질의 순서·점수·예측·불확도·분할·지표·학습 기록·포즈 바이트가 같아야 한다.
- **과학적 방법을 바꾸는 변경**(선택 규칙, 학습 프로토콜, 지문, 중복 제거 규칙 등)은 결과가 달라져도 되지만, 무엇이 왜 달라졌는지 커밋 메시지와 README에 적고, 그 행동을 검증하는 테스트를 추가한다. 이전 결과와 새 결과를 프로토콜 구분 없이 섞지 않는다.
- 건너뛴(skip) 테스트를 통과로 세지 않는다. 커버리지를 보고할 때 skip 요약을 함께 확인한다.

## 자원 정책 (이 서버의 운영 규칙)

- 모든 계산 검증과 실험은 CPU 9개 이하, 물리 GPU 0 또는 1만 사용한다. 이 한계는 다음 위치에 하드코딩되어 있으며, 넓히려면 전부 함께 고쳐야 한다(하나라도 빠지면 그 경로만 이전 한계로 실패한다). 사용자 승인 없이 넓히지 않는다.
  - `config.ExperimentConfig.validate()` — `cpu_threads` 1–9, `gpu_ids` ⊆ {0,1}, `device` ∈ {cpu, cuda:0, cuda:1}
  - `resources.configure_resources` — 스레드 1–9, GPU ⊆ {0,1}
  - `docking.DockingOracle` — cpu 1–9
  - `store.build_library_store` — workers 1–9
  - `model.TabMRegressor`, `foundation.FoundationRegressor` — device 집합과 `CUDA_VISIBLE_DEVICES` 토큰 {0,1}
  - `scripts/prepare_receptor.py`, `scripts/build_replay.py` — `--cpu` 1–9
- `docking.workers × docking.cpu ≤ resources.cpu_threads`. 이 조건을 우회하는 병렬화를 추가하지 않는다.
- 여러 실험을 동시에 띄울 때는 `taskset` 등으로 CPU 집합을 겹치지 않게 나눈다.

## 모듈 경계와 import

- `config.py`, `model_registry.py`, `cli.py`의 최상위에서 numpy, pandas, torch, RDKit, scikit-learn, foundation 패키지를 import하지 않는다. 이들은 자원 설정 전에 실행되며, 위반하면 스레드·GPU 제한이 적용되지 않은 채 수치 라이브러리가 초기화된다.
- `cli.py`는 `configure_resources`를 호출한 **다음에** `pipeline`/`store`를 import한다.
- `pipeline.py`는 PyTorch·도킹·모델(`regressors`, `model`, `foundation`) 모듈을 각 단계 함수 안에서 import한다. 최상위 import(`store`, `data`를 통한 numpy·pandas·RDKit)를 더 늘리지 않는다.
- 점수 출처는 오라클 객체로만 추가한다: `evaluate(cid, smiles, output_dir) -> {"score", "status": "ok"|"failed", "error", "pose"}`. 새 점수 출처를 위해 파이프라인 단계를 고치지 않고 `ActiveLearningExperiment.make_oracle`을 재정의한다.
- 새 회귀 백엔드는 세 곳을 함께 고친다: `model_registry.MODEL_CATALOG`(+필요 시 `ALIASES`), `regressors._make_member`, `regressors.load_regressor`. 백엔드 객체는 `fit(X, y, groups)`, `predict(X) -> (mean, spread)`, `save(dir)`, `load(dir, device=)`, `get_model_info()`(`model_type` 포함), `history_`, `best_validation_loss_`, `train_indices_`, `validation_indices_`를 제공해야 한다. 파이프라인이 이 속성들을 직접 읽는다.
- 백엔드 파라미터 블록에서 `device`, `seed`, `random_state`, `n_jobs`, `cpu_threads`를 받지 않는다(`ModelConfig.validate_selection`이 거부). 트리 백엔드는 `warm_start`도 거부한다.
- 요청한 백엔드가 없으면 설치 안내와 함께 실패한다. 다른 백엔드로 조용히 대체하지 않는다. 도킹 엔진도 마찬가지로 다른 구현으로 대체하지 않는다.

## 설정 스키마

- 모든 설정은 `schema_version: 1` 데이터클래스(`config.py`)로 정의된다. 알 수 없는 키, 중복 YAML 키, 잘못된 타입, 모순된 자원 설정은 실패해야 한다. 이 엄격함을 완화하지 않는다.
- 설정 키를 추가·변경하면 같은 커밋에서 `ExperimentConfig.validate()` 검증, `config_full.yaml`(전체 스키마 예시), README의 해당 표를 함께 고친다.
- 경로 값(`--set` 포함)은 YAML 파일 디렉터리 기준으로 해석한다. 새 경로형 설정을 추가하면 `config.from_dict`의 경로 해석 목록에도 넣는다.

## 데이터·산출물 처리

- 현재 새 실행은 새것이거나 비어 있는 출력 디렉터리에서만 시작한다. 이미 완료된 사이클 디렉터리, 포즈 파일, 저장된 모델처럼 확정된 증거를 덮어쓰거나 지우는 코드 경로를 만들지 않는다. 스크립트는 첫 출력 전에 `provenance.validate_new_paths`로 모든 생성 경로를 점검한다.
- 수치 배열은 `allow_pickle=False`, PyTorch 가중치는 `weights_only=True`로 로드하고, 저장 시 기록한 SHA-256을 로드 시 확인한다.
- JSON은 `allow_nan=False`로 쓴다. 정의되지 않은 지표는 `null`로 기록한다.
- 라이브러리는 ID 정렬 상태를 유지해야 한다. 위치를 동점 키로 쓰는 코드(`select_indices`, `discovery_metrics`, `np.searchsorted` 조회)가 이 정렬에 의존한다.

## 제거된 기능

다음은 의도적으로 제거되었으며, 다른 이름(특히 TabM 이름)으로 되살리지 않는다: 자체 구현 MLP(`TabMNetwork`), 추론 시 Monte Carlo dropout, ODDT 기반 오케스트레이션, 기준 자세 RMSD/MCS/수소 결합 기반 학습 라벨 검열. TabM은 공식 `tabm==0.0.3`의 `TabM.make`만 사용하고 구조를 복사·재정의하지 않는다.

## 의존성

- 패키지 의존성 범위는 `pyproject.toml`에 둔다. 설치기 잠금 파일은 `requirements/install.in`에서 uv 0.11.7로 생성하며, 각 잠금 파일 첫머리의 명령을 그대로 쓴다. 의존성을 바꾸면 CPU·cu128 잠금 파일을 둘 다 다시 생성한다. 하나만 갱신하면 두 설치 프로필의 패키지 버전이 달라진다.
- 새 외부 라이브러리는 꼭 필요할 때만 추가하고, 추가 이유를 주석으로 남긴다.

## 코드 작성 규칙

- 임시·실험 코드, 디버그 출력, 자리표시 주석을 커밋하지 않는다.
- 같은 로직을 중복 구현하지 않는다. 기존 함수를 재사용하거나 리팩터링한다.
- 불필요한 새 파일을 만들지 않는다. 기존 파일과 구조에 추가한다.
- 제출하는 코드는 완전하고 실행 가능해야 하며, 요청받은 범위만 수정한다.
- `try/except`는 꼭 필요한 곳에서만, 구체적 예외 클래스로, 가장 좁은 범위에 쓴다.
- 함수·클래스에는 목적·인자·반환값을 담은 docstring을 쓴다.
- PEP 8, 줄 길이 110(`pyproject.toml`의 ruff 설정, 규칙 `E4,E7,E9,F`).
- 복잡한 로직을 한 줄로 과도하게 압축하지 않는다.
- 위험하거나 의도치 않은 동작을 추가하기 전에 그 이유를 먼저 설명한다.

## 사용자 문서

- `README.md`, `docs/reference.md`, `docs/model_backends.md`는 영어 사용자 문서다. 사용법(명령, 설정, 출력, 설치)이 바뀌면 같은 커밋에서 이 문서들을 최신으로 고친다.
- 사용자 문서는 이 세 파일 밖으로 늘리지 않는다. README는 개요·설치·사용법·API·예제·기여 지침 구성을 유지한다.
- `docs/` 아래의 나머지 한국어 문서와 디렉터리별 `AGENTS.md`는 개발 규칙·지식 문서다. `MANIFEST.in`의 `recursive-include docs *.md` 때문에 한국어 문서도 sdist에 포함되며, 연구실 내부용이라 그대로 둔다(사용자 결정).
- 번들 설정 파일(`config_*.yaml`, `examples/*.yaml`)은 항상 `validate`와 `run`이 통과하는 상태여야 한다.

## 커밋·브랜치

- `main`에 직접 커밋한다. 커밋 메시지는 한국어로 쓴다.
- `results/`, `artifacts/`, `.venv*/`는 커밋하지 않는다(Git 제외 대상).
