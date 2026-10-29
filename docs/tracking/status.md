# 현재 상태

기준 커밋: `dce8df0` (2026-10-29 16:31 KST, GNINA CNNaffinity 재채점과 TabPFN v2 기본 모델 문서 정리). 아래 최종 검증은 2026-10-29 이 커밋 그대로(매니페스트 `git_dirty: false`)에서 실행했고, 그 뒤 커밋은 문서만 바꾼다(코드 변경 없음).

## 완료

| 기능 | 상태 | 근거 |
|---|---|---|
| 엄격한 설정 스키마, `--set` 덮어쓰기, YAML 기준 경로 | 구현·검증 | `tests/test_config.py`, 전체 테스트 통과 기록 |
| CPU affinity·스레드·물리 GPU 0/1 제한, 시드 | 구현·검증 | `tests/test_resources.py`, GPU 0·1 replay 재현 기록 |
| CSV 검증·정규화·중복 대표 규칙·Morgan/feature-Morgan 지문·그룹 홀드아웃 | 구현·검증 | `tests/test_data.py` |
| 10^8 규모 라이브러리 저장소(`prepare-library`, 압축 지문, memory-map) | 구현·검증 | `tests/test_store.py`, CSV 실행과 저장소 실행의 정확 일치, 1·2 워커 저장소 바이트 동일, 895k 합성 저장소 GPU 실행 2회 동일 |
| TabM(공식 `tabm==0.0.3`) 학습·튜닝·체크포인트 | 구현·검증 | `tests/test_model.py`. 이제 기본 모델이 아니며 `model.models: [tabm]`으로 선택 |
| TabPFN v2 기본 모델(고정 체크포인트, 획득 전체 기준 특성 500개, 분포 표준편차) | 구현·검증 | `tests/test_foundation.py` 계약 테스트, 실제 가중치 테스트 5개 통과(skip 0), replay `reproduce` 통과, GPU 1 실행 2회 정확 일치, 결정 0010 |
| 설정 규칙: TabM 미선택 시 TabM 전용 값 오류, TabPFN CPU + 예산 1000 초과 거부 | 구현·검증 | `tests/test_config.py`, 번들 설정 9개 `validate`·`run` 통과 |
| RF/ET, 평균·가중·검증 가중 앙상블 | 구현·검증 | `tests/test_ensembles.py`, 3모델 replay 재현, 실제 Vina 3모델 실행(6/6 성공, 복원 예측 일치) |
| Nori·TabLDM·Google TabFM 로컬 어댑터 | 구현·검증 | 가짜 upstream 계약 테스트 + 실제 공개 가중치 테스트 3개 통과, 4모델 replay 재현 19개 검사 통과·복원 예측 48개 일치 |
| 선택 규칙 4종과 beta 스케줄 | 구현·검증 | `tests/test_acquisition.py` |
| 회귀·발견 지표 | 구현·검증 | `tests/test_evaluation.py` |
| Vina/QVina 도킹 오라클, 격리 작업 프로세스, 동시 워커 | 구현·검증 | `tests/test_docking.py`, 실제 도킹 통합 테스트 3개, 워커 1·2개 결과 정확 일치 |
| GNINA 1.3.3 CNNaffinity 재채점(`docking.rescoring: gnina`, rootless Apptainer, CPU·네트워크 차단) | 구현·검증 | 가짜 apptainer 계약 테스트(실패 예산 소모·점수 없음, 버전 불일치 예산 전 실패), 실제 GNINA 통합 테스트 통과(같은 실행 2회·워커 1·2개 정확 일치), 결정 0009 |
| 단계형 `ActiveLearningExperiment`, `ReplayOracle` | 구현·검증 | `tests/test_pipeline.py` |
| `reproduce`, `benchmark`, `compare_runs` | 구현·검증 | `tests/test_experiments.py` |
| 수용체 준비·replay fixture 생성 스크립트 | 구현·검증 | `tests/test_scripts.py`, 수용체 PDBQT 독립 반복 바이트 동일 |
| uv 고정 설치기(CPU/cu128)와 설치 검증기 | 구현·검증 | `tests/test_installation.py`, 설치 검증 보고서 |
| GitHub Actions CI (Python 3.10·3.12, 비통합 테스트·replay 재현·빌드) | 구성됨 | `.github/workflows/tests.yml` |

최종 검증 기록 (2026-10-29, 커밋 `dce8df0`, 로그는 `results/final_verification_20261029/logs/`):
- 전체 테스트(`.venv-ald-tabpfn`, `taskset -c 0,1`): 631 통과, skip 9(모두 opt-in: 실제 도킹 4, 실제 foundation 가중치 5). ruff 통과.
- 비회귀: 기준 커밋 `26ef305` 대비 9개 실행(재현성 목록 8개 + `examples/foundation.yaml`, 기준 worktree 설정 파일, `models` 미지정 설정은 `[tabm]`) `compare_runs` 모두 `passed: true`. `results/nonregression-20261029/{before,after}`.
- 실제 GNINA·도킹(`ALD_RUN_DOCKING_TESTS=1`, `ALD_GNINA_IMAGE`, `ALD_APPTAINER`, `-m integration`): 4 통과, skip 0.
- 실제 foundation 가중치(`ALD_RUN_FOUNDATION_TESTS=1`, `-m foundation`): 5 통과(tabpfn 포함), skip 0.
- `reproduce --config examples/replay.yaml`(CPU, TabPFN): 통과. `results/final_verification_20261029/reproduce_replay_cpu`.
- `examples/gpu.yaml`(TabPFN) 물리 GPU 1 실행 2회: `compare_runs` 통과(검사 27개). `results/final_verification_20261029/run_gpu_cuda1_{a,b}`.
- 번들 설정 9개 `validate` 통과, 각각 `run` 1회 완료(gpu는 cu128 환경·GPU 1, foundation은 CPU). `results/final_verification_20261029/run_*`.
- 설치 검증(`bash install.sh --test`):
  - CPU: `.venv-ald-verify/installation-checks/20261029T080010Z-akdjpfpn/report.json` — 통과, 테스트 640개, skip 6(실제 foundation 5: `--test-models`가 아니므로, 실제 GNINA 1: 이미지 환경 변수 없음)
  - CUDA 12.8(물리 GPU 1): `.venv-ald-verify-cu128/installation-checks/20261029T080408Z-uu9uh4hv/report.json` — 통과, 테스트 640개, skip 6(같은 이유)
- TabPFN 처리량(GPU 1, 151,432개 합성 저장소): `docs/engineering-notes.md`의 "TabPFN v2 측정 기준값", `results/tabpfn_throughput_20261029/`.

이전 기록(2026-09-06, TabM 기본, `bash install.sh --test-models` 481개 skip 0)은 `.venv-ald*/installation-checks/`와 `results/model_expansion_status.json`에 남아 있다. 이 보고서들은 Git 밖(환경 디렉터리, `results/`)에만 있다.

환경:
- `.venv-ald-tabpfn`(CPU), `.venv-ald-tabpfn-cu128`: TabPFN 도입 작업에서 만든 검증 환경.
- `.venv-ald-verify`, `.venv-ald-verify-cu128`: 위 설치 검증으로 만든 환경.
- 사용자의 기존 `.venv-ald`, `.venv-ald-cu128`에는 `tabpfn`이 없다. 기본 설정(`models` 미지정)은 preflight에서 실패하므로 설치기를 다시 돌리거나 위 새 환경을 쓴다. 예전 환경으로 TabM을 쓰려면 `model.models: [tabm]`을 명시한다.
- GNINA: `/data2/home/jyryu/gnina/gnina-1.3.3.sif`, Apptainer `/data2/home/jyryu/opt/apptainer/1.5.5/bin/apptainer`(설정 또는 테스트 환경 변수로 지정).

## 남은 작업 (우선순위 순)

1. **예전 호환 진입점 제거** — 사용자 결정으로 바로 제거 가능. 대상: `main.py`, `Activedocking.py`, `unified_model.py`, 루트 `__init__.py`. 함께 고칠 곳: `pyproject.toml`의 `py-modules`와 ruff `include`, `tests/test_pipeline.py`의 호환 테스트, README의 `python main.py --config ...` 안내와 ruff 명령, `docs/reference.md`의 이전 구현 이전 안내. CI의 `ruff check .`는 그대로 동작한다.
2. **중단된 실행 재개** — 계획됨. 현재는 실행이 중단되면 새 출력 디렉터리에서 처음부터 다시 실행해야 하며, 모델 복원은 예측 재현만 한다. 방향(사용자 결정): 재개는 **같은 실행 디렉터리에 이어 쓴다**. 이미 완료된 `cycle_NNN/` 디렉터리와 포즈 파일은 바꾸지 않고 뒤 사이클만 추가한다. 재개를 만들 때 지켜야 할 것: 이미 시도한 화합물(실패 포함)의 예산 소모 유지, 재개한 실행과 한 번에 끝낸 실행의 결과 정확 일치. 설계 때 정할 것: 누적 파일(`observations.csv`, `metrics.csv`, `training_history.csv`, `manifest.json`)의 갱신 방식, 중단된 사이클의 부분 산출물 처리, 현재의 "비어 있는 출력 디렉터리만 허용" 검사를 재개 명령에서 어떻게 예외로 둘지.
3. **10^8 규모 TabPFN 풀 예측 처리량** — 사용자 판단 필요. GPU 1 측정에서 10^8 풀 1회 예측 외삽이 문맥 1,000행 약 77시간, 10,000행 약 464시간이다(TabM 약 8분). 이번 작업은 측정·기록만 했고 설계는 바꾸지 않았다. 대형 풀에는 TabM·트리 백엔드를 쓰거나 풀 축소·사전 선별 같은 방법 변경이 필요하다.

## 의도적으로 범위 밖

- 양성자화·호변이성체·입체이성질체 열거
- 여러 서버·노드에 걸친 분산 실행
- 수용체 누락 잔기·루프 재건

## 차단 요소

없음.
