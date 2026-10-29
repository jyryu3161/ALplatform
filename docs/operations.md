# 운영 절차

## 전제 조건

- Linux x86_64, glibc 2.28 이상
- Bash, Git, GNU coreutils(`sha256sum`, `realpath`, `install`), `getconf`, `tar`
- uv 0.11.7이 `PATH`나 캐시에 없을 때만 `curl` 또는 `wget`
- 미리 설치된 Python, conda, sudo는 필요 없다
- 이 서버에서는 CPU 9개 이하, 물리 GPU 0·1만 사용한다

## 최초 설치

저장소 루트에서 순서대로 실행한다.

```bash
bash install.sh                       # 1) uv·Python 3.11.13·잠금 패키지 설치 + 기본 검증
source .venv-ald/bin/activate         # 2) 1이 성공한 뒤에만 존재
active-learning-docking models        # 3) 백엔드 목록 확인 (선택 런타임을 import하지 않음)
active-learning-docking validate --config examples/replay.yaml
active-learning-docking run --config examples/replay.yaml   # output_dir이 새 디렉터리여야 함
```

`install.sh`가 하는 일(순서 고정):
1. 인자 확인 → `--venv` 경로가 이미 있으면 같은 체크아웃 소유(`.ald-install-owner`)인지 확인, 아니면 중단.
2. uv 0.11.7 확보(PATH → `${XDG_CACHE_HOME:-$HOME/.cache}/active-learning-docking/uv/0.11.7` → SHA-256 검증 다운로드).
3. 관리형 Python 3.11.13 설치와 가상환경 생성(새 환경일 때만).
4. 잠금 파일의 빌드 백엔드 8종을 먼저 설치 → 나머지를 `--no-deps --no-build-isolation`으로 설치. 빌드 백엔드가 먼저 없으면 TabLDM(Hatchling)·Nori(setuptools<81) 빌드가 실패한다.
5. `uv pip check`, `installed-freeze.txt` 기록, `scripts/verify_install.py` 실행.

| 옵션 | 동작 |
|---|---|
| (없음) | CPU PyTorch 2.8.0, import 확인, TabM+RF+ET 소형 replay(성공 12개, 유한 예측 48개 확인) |
| `--test` | 위 + 실제 도킹 테스트를 켠 전체 테스트 |
| `--test-models` | `--test` + Nori·TabLDM·TabFM·TabPFN v2 공개 가중치 실제 추론·복원과 TabPFN replay 정확 재현(첫 실행 시 가중치 다운로드, TabPFN v2는 약 44 MB) |
| `--torch-backend cu128` | CUDA 12.8 PyTorch(약 3.7 GiB 추가), 사용 가능한 물리 GPU 0 또는 1 필요 |
| `--venv PATH` | 다른 환경 경로(새 경로이거나 이 체크아웃 소유 환경) |
| `--offline` | 캐시된 uv·Python·패키지(·가중치)만 사용 |

CUDA 환경은 별도 경로로 만든다:
```bash
bash install.sh --torch-backend cu128 --venv .venv-ald-cu128 --test
source .venv-ald-cu128/bin/activate
active-learning-docking run --config examples/gpu.yaml
```

검증 결과는 `<환경>/installation-checks/<타임스탬프-접미사>/report.json`(+ `runtime.log`, replay 로그·결과, 요청 시 `pytest.log`·`pytest.xml`)에 남는다. `--test`는 실제 Vina/QVina 통합 테스트가, `--test-models`는 거기에 더해 foundation 실제 테스트 5개(Nori·TabLDM·TabFM·TabPFN 추론·복원, TabPFN replay 정확 재현)가 skip되거나 실패하면 설치 검증 전체를 실패로 처리한다.

`setup_server.sh`는 `install.sh`로 인자를 그대로 넘긴다. `source activate_env.sh [PATH]`는 지정 환경, 없으면 `.venv-ald`, 그것도 없으면 예전 `.venv`를 활성화한다.

## 개발 검증 명령

커밋 전(필수):
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0,1 taskset -c 0,1 .venv-ald/bin/python -m pytest -q
.venv-ald/bin/ruff check active_learning_docking tests scripts main.py unified_model.py Activedocking.py
```

필요할 때:
```bash
# 실제 Vina/QVina 통합 테스트 (도킹 관련 변경 시 권장)
ALD_RUN_DOCKING_TESTS=1 .venv-ald/bin/python -m pytest -q tests/test_docking.py -m integration

# 실제 GNINA 재채점 통합 테스트 (재채점 관련 변경 시; GNINA 1.3.3 SIF와 Apptainer 필요)
ALD_RUN_DOCKING_TESTS=1 ALD_GNINA_IMAGE=<gnina-1.3.3.sif> ALD_APPTAINER=<apptainer 실행 파일> \
  .venv-ald/bin/python -m pytest -q tests/test_docking.py tests/test_pipeline.py -m integration

# 실제 foundation 가중치 테스트 (누락 가중치 다운로드)
ALD_RUN_FOUNDATION_TESTS=1 .venv-ald/bin/python -m pytest -q tests/test_foundation.py -m foundation
ALD_RUN_FOUNDATION_TESTS=1 .venv-ald/bin/python -m pytest -q tests/test_foundation.py -m foundation -k nori

# 새 프로세스 두 번 실행 후 정확 비교
active-learning-docking reproduce --config examples/replay.yaml --output results/reproducibility

# 시드 쌍 전략 비교
active-learning-docking benchmark --config examples/benchmark.yaml --output results/benchmark \
  --seeds 17 42 73 --strategies random greedy lcb thompson

# 휠/sdist 빌드
.venv-ald/bin/python -m build
```

`reproduce`·`benchmark`·`prepare-library`의 `--output`은 새 경로여야 한다. 구조 변경 전후 비교는 변경 전 실행 디렉터리를 먼저 만들어 둔 뒤 `active_learning_docking.experiments.compare_runs(before, after)`로 한다.

## 실험 실행

```bash
active-learning-docking run --config CONFIG.yaml [--set section.key=value ...]
python -m active_learning_docking run --config CONFIG.yaml      # 같은 동작
```

- 첫 인자가 `-`로 시작하면(`--help` 제외) `run`이 생략된 것으로 본다.
- `--set` 값은 YAML 스칼라로 해석되고, 경로는 YAML 파일 디렉터리 기준이다. 예: `--set output_dir=../results/x`는 설정 파일 위치 기준이다.
- 오류(`ValueError`, `OSError`, `RuntimeError`, `ImportError`, YAML 오류)는 `Error: ...`를 stderr에 쓰고 종료 코드 1.

| 설정 파일 | 용도 |
|---|---|
| `examples/replay.yaml` | 48개 합성 점수 replay, 가장 빠른 동작 확인 |
| `config_basic.yaml` | 최소 합성 replay |
| `config_full.yaml` | 전체 스키마 |
| `config_ensemble.yaml` | TabPFN + RF + ET, 검증 가중치 |
| `examples/foundation.yaml` | TabPFN + Nori + TabLDM + TabFM (Nori·TabLDM·TabFM 런타임 필요) |
| `config_autotune.yaml` | TabM 시드 하이퍼파라미터 탐색 |
| `examples/docking.yaml` | 실제 Vina 도킹 (빠른 실행용 설정) |
| `examples/gpu.yaml` | 물리 GPU 선택 |
| `examples/benchmark.yaml` | 실제 Vina 점수 fixture로 전략 비교 |

## 데이터 준비

**대용량 라이브러리 저장소** (10^6개를 넘는 풀):
```bash
active-learning-docking prepare-library --config examples/docking.yaml --output results/candidate_store
active-learning-docking run --config examples/docking.yaml \
  --set data.path=../results/candidate_store --set output_dir=../results/docking_store
```
- 워커 수 = `resources.cpu_threads`, `--chunk-size`(기본 20,000)는 워커 작업당 CSV 레코드 수.
- 저장소를 만든 설정과 실행 설정의 지문·중복 제거·열 이름(replay면 점수 열)이 같아야 한다.
- 대형 저장소는 `evaluation.holdout_fraction: 0`, `evaluation.save_predictions: false`, `model.inference_batch_size: 8192` 정도와 CUDA 장치를 쓴다.
- 디스크: 2,048비트 기준 10^8개에 지문 25.6 GB + SMILES·ID 10–15 GB, 구축 RAM 4–6 GB, 실행 RAM 사이클당 약 6 GB + PyTorch.

**수용체 준비** (새 출력 경로 필수):
```bash
python scripts/prepare_receptor.py --input inputs/protein.pdb \
  --output results/preparation/receptor.pdbqt --repair-missing-atoms --ph 7.4 --seed 42 --cpu 1
```
`--repair-missing-atoms`는 누락 곁사슬·말단 원자를 재건하고 수소를 다시 붙인다. 누락 잔기는 재건하지 않는다. 이미 완전하고 적절히 수소화된 구조에는 생략한다.

**replay 점수 fixture 재생성**:
```bash
python scripts/build_replay.py --output artifacts/replay_repeat/docking_scores.csv \
  --artifacts artifacts/replay_repeat/docking --count 32 --seed 42 --cpu 1 --exhaustiveness 1
```
종료 코드 1이 정상(성공 31개 + 염 실패 1개).

**합성 replay 예제 재생성**: `python examples/make_demo.py`

## 환경 변수

| 변수 | 역할 | 범위 |
|---|---|---|
| `ALD_RUN_DOCKING_TESTS=1` | 실제 Vina/QVina 통합 테스트 활성화 | 테스트 실행 시에만 |
| `ALD_RUN_FOUNDATION_TESTS=1` | 실제 foundation 가중치 테스트 활성화(다운로드 발생) | 테스트 실행 시에만 |
| `ALD_GNINA_IMAGE`, `ALD_APPTAINER` | 실제 GNINA 재채점 통합 테스트용 SIF·Apptainer 경로. 없으면 그 테스트는 skip | 테스트 실행 시에만 |
| `CUDA_VISIBLE_DEVICES` | 상속되면 허용 GPU와 교집합. 숫자가 아닌 값이면 GPU 실행 불가 | 실행 전 셸 |
| `XDG_CACHE_HOME` | uv 캐시 위치 | 설치 시 |
| `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` | 테스트 시 1로 지정. 실험 실행 시에는 `configure_resources`가 `cpu_threads`로 덮어씀 | 테스트 셸 |

실험 실행 중 `configure_resources`가 `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, 스레드 변수, `OMP_DYNAMIC=FALSE`를 직접 설정한다. 이 값을 셸에서 따로 맞출 필요는 없다.

## 배포

배포 절차는 없다. 연구실 내부에서 소스 체크아웃 + `install.sh`로 사용한다. 패키지 빌드(`python -m build`)는 휠/sdist가 만들어지는지 확인하는 용도다.

## 실험 보관

보고하는 실험은 실행 디렉터리 전체(매니페스트·해시 포함), 사용한 잠금 파일, 환경 freeze, 정확한 커밋을 함께 보관한다. `results/`와 `artifacts/`는 Git에 올리지 않는다.
