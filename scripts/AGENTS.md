# scripts

## 범위

패키지 밖에서 단독 실행하는 세 스크립트:
- `prepare_receptor.py` — 단백질 PDB → 엄격한 Meeko 타이핑의 수용체 PDBQT + 출처 기록(provenance JSON, 중간 PDB, 준비 로그)
- `build_replay.py` — 후보 CSV 앞부분을 실제로 도킹해 replay 점수 fixture(CSV + provenance JSON + 포즈 아티팩트) 생성
- `verify_install.py` — `install.sh`가 호출하는 설치 검증(import, 장치, 소형 replay, 선택적 테스트 실행, `report.json`)

범위가 아닌 것: 능동 학습 실험 실행(패키지 CLI), 설치 자체(`install.sh`), 번들 fixture 파일의 직접 수정. 이 스크립트에 실험 로직(모델 학습·질의 선택)을 넣지 않는다.

## 불변 조건

- 출력은 항상 새 경로다. 첫 출력이나 상위 디렉터리를 만들기 **전에** `active_learning_docking.provenance.validate_new_paths`로 모든 생성 파일·디렉터리를 점검한다. 기존 파일·링크(끊긴 링크 포함), 입력과 같은 실제 경로, 생성 경로끼리의 충돌은 인자 오류로 끝나고 아무것도 만들지 않는다.
- `build_replay.py`의 기본 출력은 `examples/data/docking_scores.csv`지만 그 파일이 이미 있으므로 기본값으로는 실패한다. 재생성은 항상 새 `--output`/`--artifacts`로 한다. 출력 CSV·provenance는 `--artifacts/poses` 아래에 둘 수 없다.
- `build_replay.py`는 ID 정렬·대표 규칙을 적용한 라이브러리를 **원래 입력 순서(파일, 레코드 번호)**로 다시 정렬해 앞 `--count`개를 고른다. 점수가 선택에 영향을 주지 않는다. 실패한 화합물은 점수 없이 기록하고, 실패가 하나라도 있으면 provenance를 다 쓴 뒤 종료 코드 1이다(번들 입력에서는 염 1개 실패가 정상).
- 도킹은 패키지의 `DockingOracle`을 그대로 쓴다. 별도 도킹 구현을 만들지 않는다. provenance에는 실제 import된 도킹 모듈의 해시를 기록한다.
- `prepare_receptor.py`는 `--repair-missing-atoms`가 있을 때만 PDBFixer로 누락 곁사슬·말단 원자를 재건하고 수소를 다시 붙인다(OpenMM Reference CPU 플랫폼, 시드 고정). 누락 잔기는 재건하지 않고, 체인·물·보조인자·맞지 않는 잔기를 조용히 지우지 않는다. 기존 중원자 좌표는 그대로 둔다. Meeko 준비는 `python -m meeko.cli.mk_prepare_receptor`를 `--timeout` 안에서 실행한다.
- CPU 인자는 1–9. pH ∈ [0, 14], 시드 ∈ [1, 2³¹).
- `verify_install.py`의 `--test`는 실제 Vina/QVina 통합 테스트가, `--test-models`는 거기에 더해 foundation 실제 추론·복원 테스트가 **skip 없이 통과**해야 성공이다. skip을 통과로 바꾸지 않는다. 검증 디렉터리는 매번 새로 만든다.

## 구현 패턴

- 인자는 `argparse`, 오류는 `parser.error`(종료 2) 또는 예외. 해시는 SHA-256.
- provenance JSON에는 입력·출력 해시, 사용한 패키지 버전, 실행 명령, 시드, 환경 정보를 넣는다.
- 패키지 내부 함수(`provenance`, `data`, `docking`)를 import해 재사용한다.

## 테스트 지침

- `tests/test_scripts.py`가 다루는 것: 모든 생성 파일의 사전 점검, 생성 경로와 입력의 별칭 충돌, 생성 경로끼리의 충돌, 디렉터리 입력 거부(평가 전), 새 아티팩트 디렉터리 안의 출력과 import된 오라클 해시 기록. 새 출력 경로를 추가하면 같은 형태의 사전 점검 테스트를 추가한다.
- `tests/test_installation.py`가 다루는 것: 상속 CUDA 마스크, 잘못된 torch 빌드, 소형 replay 결과 검증, 실패 시 증거 보존, 요청한 통합 테스트만 활성화, 필수 통합 테스트 skip 금지, 설치기 인자·소유되지 않은 환경 거부.
- 수용체 준비의 반복 바이트 동일성은 자동 테스트가 아니라 수동 재생성으로 확인한 기록이다. 준비 로직을 바꾸면 같은 명령을 새 경로에 두 번 실행해 PDBQT 해시를 비교한다.
- 실제 도킹이 필요한 경로는 `ALD_RUN_DOCKING_TESTS=1`일 때만 실행된다.
