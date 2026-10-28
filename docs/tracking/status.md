# 현재 상태

기준 커밋: `8090e94` (2026-09-06 20:33 KST, 로컬 회귀 앙상블과 uv 통합 설치·검증 추가). 마지막 전체 검증은 같은 날 이 커밋보다 먼저(18:42, 19:10 KST) 실행되었으므로, 커밋된 코드 그대로에서 통과했는지는 기록으로 확인되지 않는다. 커밋 후 작업 트리는 변경 없음.

## 완료

| 기능 | 상태 | 근거 |
|---|---|---|
| 엄격한 설정 스키마, `--set` 덮어쓰기, YAML 기준 경로 | 구현·검증 | `tests/test_config.py`, 전체 테스트 통과 기록 |
| CPU affinity·스레드·물리 GPU 0/1 제한, 시드 | 구현·검증 | `tests/test_resources.py`, GPU 0·1 replay 재현 기록 |
| CSV 검증·정규화·중복 대표 규칙·Morgan/feature-Morgan 지문·그룹 홀드아웃 | 구현·검증 | `tests/test_data.py` |
| 10^8 규모 라이브러리 저장소(`prepare-library`, 압축 지문, memory-map) | 구현·검증 | `tests/test_store.py`, CSV 실행과 저장소 실행의 정확 일치, 1·2 워커 저장소 바이트 동일, 895k 합성 저장소 GPU 실행 2회 동일 |
| TabM(공식 `tabm==0.0.3`) 학습·튜닝·체크포인트 | 구현·검증 | `tests/test_model.py` |
| RF/ET, 평균·가중·검증 가중 앙상블 | 구현·검증 | `tests/test_ensembles.py`, 3모델 replay 재현, 실제 Vina 3모델 실행(6/6 성공, 복원 예측 일치) |
| Nori·TabLDM·Google TabFM 로컬 어댑터 | 구현·검증 | 가짜 upstream 계약 테스트 + 실제 공개 가중치 테스트 3개 통과, 4모델 replay 재현 19개 검사 통과·복원 예측 48개 일치 |
| 선택 규칙 4종과 beta 스케줄 | 구현·검증 | `tests/test_acquisition.py` |
| 회귀·발견 지표 | 구현·검증 | `tests/test_evaluation.py` |
| Vina/QVina 도킹 오라클, 격리 작업 프로세스, 동시 워커 | 구현·검증 | `tests/test_docking.py`, 실제 도킹 통합 테스트 3개, 워커 1·2개 결과 정확 일치 |
| 단계형 `ActiveLearningExperiment`, `ReplayOracle` | 구현·검증 | `tests/test_pipeline.py` |
| `reproduce`, `benchmark`, `compare_runs` | 구현·검증 | `tests/test_experiments.py` |
| 수용체 준비·replay fixture 생성 스크립트 | 구현·검증 | `tests/test_scripts.py`, 수용체 PDBQT 독립 반복 바이트 동일 |
| uv 고정 설치기(CPU/cu128)와 설치 검증기 | 구현·검증 | `tests/test_installation.py`, 설치 검증 보고서 |
| GitHub Actions CI (Python 3.10·3.12, 비통합 테스트·replay 재현·빌드) | 구성됨 | `.github/workflows/tests.yml` |

최종 검증 기록 (`bash install.sh --test-models` 보고서):
- CPU: `.venv-ald/installation-checks/20260906T094244Z-5bth9zrt/report.json` — 통과, 테스트 481개, skip 0 (실제 Vina/QVina, 실제 foundation 가중치 포함)
- CUDA 12.8: `.venv-ald-cu128/installation-checks/20260906T101009Z-tmfmqey1/report.json` — 통과, 테스트 481개, skip 0
- ruff 통과와 휠/sdist 빌드 확인은 `results/model_expansion_status.json`에 기록되어 있다.

이 보고서들은 Git 밖(환경 디렉터리, `results/`)에만 있다. 테스트를 다시 돌리면 새 보고서가 생기며, 그 결과로 이 표를 갱신한다.

## 남은 작업 (우선순위 순)

1. **예전 호환 진입점 제거** — 사용자 결정으로 바로 제거 가능. 대상: `main.py`, `Activedocking.py`, `unified_model.py`, 루트 `__init__.py`. 함께 고칠 곳: `pyproject.toml`의 `py-modules`와 ruff `include`, `tests/test_pipeline.py`의 호환 테스트, README의 `python main.py --config ...` 안내와 ruff 명령, `docs/reference.md`의 이전 구현 이전 안내. CI의 `ruff check .`는 그대로 동작한다.
2. **중단된 실행 재개** — 계획됨. 현재는 실행이 중단되면 새 출력 디렉터리에서 처음부터 다시 실행해야 하며, 모델 복원은 예측 재현만 한다. 방향(사용자 결정): 재개는 **같은 실행 디렉터리에 이어 쓴다**. 이미 완료된 `cycle_NNN/` 디렉터리와 포즈 파일은 바꾸지 않고 뒤 사이클만 추가한다. 재개를 만들 때 지켜야 할 것: 이미 시도한 화합물(실패 포함)의 예산 소모 유지, 재개한 실행과 한 번에 끝낸 실행의 결과 정확 일치. 설계 때 정할 것: 누적 파일(`observations.csv`, `metrics.csv`, `training_history.csv`, `manifest.json`)의 갱신 방식, 중단된 사이클의 부분 산출물 처리, 현재의 "비어 있는 출력 디렉터리만 허용" 검사를 재개 명령에서 어떻게 예외로 둘지.

## 의도적으로 범위 밖

- 양성자화·호변이성체·입체이성질체 열거
- 여러 서버·노드에 걸친 분산 실행
- 수용체 누락 잔기·루프 재건

## 차단 요소

없음.
