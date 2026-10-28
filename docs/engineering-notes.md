# 실무 노트

## 실행 환경

**공유 서버에서 시간이 터무니없이 측정됨**
- 증상: `unpackbits` 같은 단순 연산이 수 초 걸리거나 같은 작업 시간이 20배씩 흔들린다.
- 원인: 같은 서버의 다른 Python 작업이 만든 메모리 압박(부하 14 안팎에서 관찰). 할당이 멈춘다.
- 대응: 코드를 "최적화"하기 전에 `free -g`로 MemFree를 확인하고, 여유가 돌아오면 다시 잰다.

**환경마다 테스트 결과·패키지 버전이 다르거나 `install.sh --venv .venv`가 거부됨**
- 원인: 저장소에 성격이 다른 환경이 공존한다. `.venv`는 anaconda Python 3.10을 가리키는 예전 개발 환경(잠금 없음, foundation 백엔드 없음), `.venv-ald`는 `install.sh`가 만든 uv 관리 Python 3.11.13 잠금 환경(ruff·pytest·CLI 포함), `.venv-ald-cu128`은 그 CUDA 빌드다. 설치기는 conda 인터프리터와 소유 표시(`.ald-install-owner`)가 없는 기존 환경을 거부한다.
- 대응: 검증·실험은 `.venv-ald/bin/python`(GPU면 `.venv-ald-cu128`)으로 한다. `source activate_env.sh`는 인자가 없으면 `.venv-ald`를, 그것이 없을 때만 `.venv`를 고르므로 어느 환경이 활성화됐는지 `which python`으로 확인한다.

**CI 통과와 설치 환경 통과는 다른 증거임**
- GitHub Actions는 Python 3.10·3.12에서 `pyproject.toml` 범위로 `[docking,tuning,dev]`만 설치해 `ruff check .`, `pytest -m 'not integration and not gpu'`, replay `reproduce`, 빌드를 돌린다. foundation 백엔드와 실제 도킹은 CI에서 검증되지 않는다.
- 설치기는 Python 3.11.13 + 잠금 파일로 전 백엔드를 설치한다. 한쪽 통과가 다른 쪽 통과를 뜻하지 않는다.

## 자원 설정

**"configure resources before CUDA initialization; start a new process"**
- 원인: 같은 프로세스에서 이미 CUDA가 초기화된 뒤(노트북, 이전 실험, 직접 torch 사용) 다른 `CUDA_VISIBLE_DEVICES`로 `configure_resources`를 호출했다.
- 대응: 새 프로세스에서 실행한다. 한 프로세스에서 장치가 다른 실험을 연달아 돌리지 않는다. `reproduce`/`benchmark`가 매번 새 인터프리터를 띄우는 이유다.

**"configure resources before starting parallel PyTorch work"**
- 원인: PyTorch의 interop 스레드 수는 병렬 작업이 시작된 뒤 바꿀 수 없다.
- 대응: 수치 연산 전에 `configure_resources`를 한 번만 호출한다.

**`taskset`으로 CPU를 더 줘도 테스트가 빨라지지 않음**
- 원인: `tests/conftest.py`가 affinity를 허용 집합의 앞 2개 CPU로 줄이고 스레드를 1로 고정한다.
- 대응: 정상 동작이다. 스케줄러 환경에서는 `taskset -c`에 할당받은 CPU 중 2개를 넣는다.

**"physical GPU 1 is excluded by the resource policy or inherited CUDA mask"**
- 원인: `resources.device: cuda:1`은 물리 GPU 1번을 뜻하고, 상속받은 숫자형 `CUDA_VISIBLE_DEVICES`와 교집합을 취한다. 셸에 `CUDA_VISIBLE_DEVICES=0`이 남아 있으면 GPU 1이 빠진다. 숫자가 아닌 마스크(UUID 등)가 상속되면 GPU 실행 자체가 거부된다.
- 대응: 셸의 `CUDA_VISIBLE_DEVICES`를 지우거나 `0,1`로 둔다. GPU 0은 다른 사용자와 공유되므로 실행 전에 `nvidia-smi`로 사용 중인지 확인한다.

## 재현성

**구조 변경 후 `compare_runs`가 실패함**
- 먼저 의심할 것: (1) `model.inference_batch_size`가 두 실행에서 다름 — 검증 손실·예측 마지막 자릿수가 바뀐다. (2) 장치나 PyTorch/CUDA 버전이 다름 — 별개 환경이다. (3) 한쪽에만 있는 사이클 CSV/`metrics.json` — 파일 목록 차이도 실패로 친다.
- 비교 절차: 변경 전 커밋(예: `git worktree`로 별도 체크아웃)에서 아래 실행을 `B=<scratch>/before`로, 변경 후 작업 트리에서 같은 실행을 `B=<scratch>/after`로 한다. `output_dir`은 절대 경로로 덮어써서 설정 파일 위치와 무관하게 한다.
  ```bash
  R=active-learning-docking
  $R run --config examples/replay.yaml   --set output_dir=$B/replay
  $R run --config config_basic.yaml      --set output_dir=$B/basic
  $R run --config config_ensemble.yaml   --set output_dir=$B/ensemble
  $R run --config config_autotune.yaml   --set output_dir=$B/autotune
  $R run --config examples/replay.yaml   --set output_dir=$B/scaffold_thompson \
     --set evaluation.split=scaffold --set learning.strategy=thompson
  $R run --config examples/gpu.yaml      --set output_dir=$B/gpu1 --set resources.device=cuda:1
  $R run --config examples/docking.yaml  --set output_dir=$B/docking_w1
  $R run --config examples/docking.yaml  --set output_dir=$B/docking_w2 --set docking.workers=2
  ```
  그 다음 각 이름마다 `python -c "from active_learning_docking.experiments import compare_runs; print(compare_runs('<before>/NAME', '<after>/NAME')['passed'])"`가 `True`여야 한다. foundation 경로를 바꿨다면 `examples/foundation.yaml`도 추가한다.

**예전 실행과 열 하나가 다름**
- `acquisition_beta` 열이 추가되기 전 실행과는 그 열만 다르다. 중복 구조의 대표 ID 규칙(사전순 최소 ID)이 바뀌기 전 v2 실행과는 중복 구조가 있는 라이브러리에서 선택이 달라질 수 있다. `artifacts/`의 이전 증거와 번들 fixture는 재생성하지 않았다.

**매니페스트에 `git_dirty: true`**
- 커밋되지 않은 변경이 있는 작업 트리에서 실행했다. 보고용 실험은 커밋 후 실행해야 소스와 결과가 연결된다.

## 대용량 라이브러리

**측정 기준값** (8 워커, 코어 0–7, 합성 fragment 라이브러리)
- `prepare-library`: 입력 200만–235만 행 → 895,315개 수용, 1분 43초–2분 11초, 최대 RSS 1.3–1.9 GB.
- 895k 풀 도킹 모드 사이클의 전체 풀 예측: L40(물리 GPU 1)에서 `inference_batch_size: 8192`로 약 4초. 6질의 실행 전체 27초, RSS 1.4 GB.
- 선형 외삽: 10^8 저장소 구축 약 1.5시간, 사이클당 풀 예측 약 8분. 정렬 재작성 단계가 임시 파일을 무작위로 읽으므로 SSD를 쓴다.
- 이 값보다 크게 느리면 코드 회귀 전에 메모리 압박부터 확인한다.

**저장소로 실행했는데 시작하자마자 실패**
- 원인: 저장소 구축 때의 지문 종류·반경·비트 수, `deduplicate`, `invalid_policy`, ID/SMILES 열 이름, (replay라면) 점수 열이 현재 설정과 다르다. 또는 파일 체크섬이 맞지 않는다(부분 복사·손상).
- 대응: 설정을 저장소에 맞추거나 저장소를 다시 만든다. 체크섬 오류면 저장소를 다시 복사하거나 다시 구축한다.

**저장소 실행이 첫 사이클 전에 오래 멈춰 있음**
- 원인: 파이프라인은 저장소를 열 때마다 `library.json`에 기록된 모든 파일의 SHA-256을 다시 계산한다(검증 기본값). 10^8개 2,048비트 저장소면 지문·문자열 열 합계 약 35–40 GB를 읽는다. 매니페스트도 `library.json` 해시를 기록한다.
- 대응: 정상 동작이며 손상 저장소를 질의 전에 잡기 위한 비용이다. 시간 추정에 디스크 순차 읽기 시간을 더한다. 검증을 끄는 경로를 파이프라인에 추가하는 것은 무결성 정책 변경이다.

**저장소 + 홀드아웃이 매우 느림**
- 원인: 홀드아웃 분할은 모든 구조를 한 번 파싱한다(정규 SMILES 또는 골격). 10^8개에서는 큰 비용이다.
- 대응: 대형 저장소는 `holdout_fraction: 0`. 도킹 모드는 어차피 필수다.

**메모리를 많이 쓰는 백엔드**
- 저장소는 풀 지문을 압축 상태로 두지만, 트리 백엔드는 획득 학습 행을 float32로 풀어서 들고 있고 foundation 백엔드는 문맥 행을 메모리에 올린다. TabM 처리량 측정값은 다른 백엔드에 적용되지 않는다. 풀을 키우기 전에 대표 부분 집합으로 시간·메모리를 잰다.

## 도킹

**실행 전체가 `FileExistsError`로 중단됨**
- 원인: 화합물 출력 디렉터리(`cycle_*/poses/<slug>_<hash>`)가 이미 있고 비어 있지 않거나 링크다. 이것은 화합물 단위 실패가 아니라 증거 보호를 위한 실험 중단이다.
- 대응: 재시도는 항상 새 출력 위치에서 한다.

**`scripts/build_replay.py`가 종료 코드 1로 끝남**
- 정상이다. 번들 후보 앞 32개 중 분리된 염(Z8462127578)이 리간드 준비에서 실패하도록 되어 있다. 결과는 성공 31개 + 구조화된 실패 1개이며, 보고서를 확인한다.

**`docking.py`를 고쳤는데 도킹 동작이 바뀌지 않음**
- 원인: 도킹 작업 프로세스는 호출한 쪽이 import한 `docking.py`의 절대 경로를 실행한다. 휠로 설치한 환경이면 설치된 사본이 실행된다.
- 대응: 편집 가능 설치(`-e .`, 설치기 기본) 환경에서 실행하고, 매니페스트의 `package_files_sha256`에서 `docking.py` 해시가 작업 트리 파일과 같은지 확인한다.

**`docking.executable: qvina2.1`인데 "QVina executable is missing or not executable"**
- 원인: `docking.executable`은 설정 경로로 취급되어 YAML 디렉터리 기준 절대 경로로 바뀐 뒤 검사된다. 이름만 주면 PATH를 찾지 않는다.
- 대응: 실행 파일 경로를 명시한다(예: 저장소 동봉 `bin/qvina2.1`을 YAML 위치 기준 상대 경로로).

**시간 초과**
- `docking.timeout`은 리간드 준비와 모든 형태의 엔진 실행을 합친 시간이다. 시간 초과 시 POSIX 프로세스 그룹 전체를 SIGKILL로 정리하고 실패로 기록한다(예산 소모). 형태 수·exhaustiveness를 올리면 timeout도 함께 올린다.

## 모델 백엔드

**TabFM을 골랐는데 `lcb`와 `greedy` 결과가 같음**
- TabFM 공개 회귀 API는 점 예측만 제공해 spread가 항상 0이다. 단독 사용 시 탐색 항이 사라진다. 앙상블에서는 다른 모델과의 불일치로 spread가 생긴다.

**TabFM 옵션이 가중치 로드 전에 거부됨**
- TabFM 1.0.1 PyTorch 회귀기에는 문맥/KV 캐시가 없다. `cache_context`, `maybe_quantize_kv_cache`, `keep_cache_on_device`는 거부된다. 로컬 체크포인트는 `config.json`과 `model.safetensors`가 있는 디렉터리(또는 `regression/` 하위를 가진 부모)여야 하고, `pytorch_model.bin`은 지원하지 않는다. 계산 dtype은 CPU·CUDA 모두 bfloat16이다.

**foundation 백엔드가 "needs at least one varying fingerprint feature" 또는 "At least two acquired labeled samples" 오류로 실패**
- 원인: 선택된 문맥 행의 지문 비트가 하나도 변하지 않거나 문맥 행이 2개 미만이다.
- 대응: 초기 표본을 늘리거나, 더 다양한 분자를 획득하거나, `max_context_samples`를 늘린다.

**포레스트 체크포인트가 다른 환경에서 로드되지 않음**
- 포레스트는 학습 배열을 저장했다가 로드할 때 결정적으로 다시 학습한다. 저장 당시와 같은 scikit-learn 버전이 필요하다.

**foundation 체크포인트 복원**
- 공개 체크포인트의 원래 캐시 경로가 없으면 저장된 고정 revision을 현재 캐시에서 다시 받아 원래 해시와 대조한다. 명시적 로컬 체크포인트는 기록된 경로에 그대로 있어야 한다.

**검증 가중치 앙상블이 초반 사이클에서 실패**
- 획득 관측의 분자 그룹이 적으면 가중치 검증 분할을 만들 수 없다(신경망/foundation 포함 시 그룹 3개 이상). 초기 표본을 키우거나 `mean` 전략으로 시작한다. 그룹이 적을 때 가중치 자체도 불안정하다.

## 반복 작업 체크리스트

**새 회귀 백엔드 추가**
1. `model_registry.MODEL_CATALOG`에 이름·구현·설치 안내 추가 → `active-learning-docking models` 출력으로 확인.
2. `regressors._make_member`와 `regressors.load_regressor`에 분기 추가.
3. 백엔드 객체가 `fit/predict/save/load/get_model_info`와 `history_`, `best_validation_loss_`, `train_indices_`, `validation_indices_`를 제공 → 단독 실행과 `[tabm, 새 백엔드]` 앙상블 실행이 끝까지 도는지 확인.
4. 무거운 런타임이면 가짜 upstream 클래스로 계약 테스트를 쓰고, 실제 가중치 테스트는 환경 변수로 opt-in → 기본 테스트에서 skip 표시 확인.
5. README·`docs/model_backends.md`의 표 갱신.

**설정 키 추가**
1. 데이터클래스 필드 + `validate()` 조건 → 잘못된 값이 실패하는 테스트.
2. 경로형이면 `from_dict` 경로 해석에 추가 → 상대 경로가 YAML 디렉터리 기준으로 풀리는지 확인.
3. `config_full.yaml`과 README 표 갱신 → `active-learning-docking validate --config config_full.yaml` 통과 확인.

**의존성 변경**
1. `requirements/install.in` 수정 → 두 잠금 파일 첫머리의 `uv pip compile` 명령을 각각 그대로 실행(uv 0.11.7).
2. `bash install.sh --venv <새 경로> --test`로 새 환경 검증 → `<환경>/installation-checks/<run>/report.json` 확인.
