# 미해결 문제

## 번들 수용체·기준 리간드의 출처와 재배포 권리 미확인

- **현상**: `inputs/protein.pdb`, `inputs/reference.pdb`와 이로부터 만든 `examples/data/receptor.pdbqt`, `examples/data/docking_scores.csv`가 저장소와 sdist에 포함되어 GitHub 원격(`jyryu3161/ALplatform`)으로 올라가지만, 실험적 출처(표적 단백질, 체인·보조인자 선택)와 재배포 권리가 확인되지 않았다. 유일한 단서는 `inputs/reference.pdb` 첫 줄의 `COMPND    8VF6`(PDB 항목 번호 형식)이며, 이것만으로는 수용체가 같은 항목에서 왔는지 정할 수 없다.
- **영향**: 이 파일로 얻은 결과(실제 Vina 점수 fixture, 전략 비교표)는 소프트웨어 동작 확인 이상으로 해석할 수 없다. 원격 저장소가 공개되거나 외부에 배포되면 권리 문제가 생길 수 있다.
- **지금 못 푸는 이유**: 출처 정보가 코드·이력·사용자 기억 어디에도 없다. 원본 파일을 받은 경로를 찾아야 한다.
- **가능한 접근**: 원본 획득 경로를 찾아 출처·라이선스를 `examples/data/receptor.provenance.json`과 사용자 문서에 기록하거나, 출처가 명확한 공개 구조로 fixture를 교체하고 `scripts/prepare_receptor.py`·`scripts/build_replay.py`로 다시 생성한다(이 경우 fixture 기반 기록은 모두 새로 만들어야 한다).

## CI가 지원 설치 환경과 foundation 백엔드를 검증하지 않음

- **현상**: GitHub Actions는 Python 3.10·3.12에서 `pyproject.toml` 범위 의존성 `[docking,tuning,dev]`로만 검사한다. 지원 설치 환경(Python 3.11.13 + 잠금 파일), Nori·TabLDM·TabFM 런타임, 실제 도킹 통합 테스트는 CI에 없다.
- **영향**: 잠금 파일 갱신이나 foundation 어댑터 변경으로 생긴 회귀가 CI를 통과한다. 반대로 범위 의존성의 새 버전 때문에 CI만 깨질 수도 있다.
- **지금 못 푸는 이유**: foundation 검증에는 약 7.2 GB 가중치와 큰 메모리가 필요하고, CI 구성 변경은 CI 자원·비용에 대한 결정이 필요하다.
- **가능한 접근**: CI에 잠금 파일 기반 Python 3.11.13 작업을 추가(가중치 없이 가짜 upstream 계약 테스트까지), 실제 가중치·도킹 테스트는 이 서버에서 `bash install.sh --test-models`로 주기적으로 돌린다.

## 지원하지 않는 설치 파일이 저장소와 sdist에 남아 있음

- **현상**: `environment.yml`, `environment-gpu.yml`(conda, Python 3.11, `[docking,tuning,dev]`), `requirements.txt`(`-e .[docking,tuning,dev]`)가 남아 있고 `MANIFEST.in`이 sdist에 포함한다. 사용자 문서는 이 파일들을 언급하지 않으며, 설치기는 conda 인터프리터를 거부한다.
- **영향**: 이 파일로 만든 환경에는 foundation 백엔드와 수용체 준비 도구가 없고 버전이 고정되지 않아, 같은 설정을 돌려도 지원 환경과 결과가 다를 수 있다.
- **지금 못 푸는 이유**: 삭제할지, 지원 경로로 문서화할지 사용자 결정이 필요하다.
- **가능한 접근**: 삭제하고 `MANIFEST.in`에서 빼거나, 사용자 문서에 "잠금 없는 최소 설치"로 명시한다.

## 동봉 QVina 실행 파일의 출처 미기록

- **현상**: `bin/qvina2.1`(정적 ELF, SHA-256 `bc1f908869181d17b48c9a9b4c2a28cce5264ccf3053e15bce32bb5baf83b43d`)이 Git과 sdist에 포함되고 QVina 통합 테스트·설치 검증(`--test`, `--test-models`)의 필수 의존성인데, 빌드 출처·버전·라이선스가 어디에도 기록되어 있지 않다.
- **영향**: QVina 결과를 보고할 때 엔진 출처를 인용할 수 없고, 파일이 바뀌어도 원본과 대조할 기준이 없다. 저장소 배포 시 재배포 조건을 확인할 수 없다.
- **지금 못 푸는 이유**: 출처 정보가 코드·이력에 없고 사용자 확인이 필요하다.
- **가능한 접근**: 원본 릴리스를 찾아 URL·버전·라이선스·해시를 기록 파일로 남기거나, 실행 파일을 저장소에서 빼고 설치 단계에서 검증된 다운로드로 바꾼다.

## 직접 API 호출 시 수치 라이브러리가 자원 제한보다 먼저 로드됨

- **현상**: `from active_learning_docking.pipeline import run_experiment` 후 `run_experiment(config)`를 `runtime` 없이 호출하면, import 시점에 `pipeline` → `store` → `data` 사슬로 numpy·pandas·RDKit이 먼저 로드되고 그 뒤에 `configure_resources`가 affinity·스레드 제한을 적용한다. CLI 경로(`run`)는 순서가 맞다. `pipeline.py` 모듈 docstring도 RDKit을 단계 안에서만 import한다고 적고 있어 실제와 다르다.
- **영향**: 직접 호출 경로에서 BLAS/OpenMP 스레드가 제한 전 설정으로 만들어져 CPU 9개 정책을 넘을 수 있다(`threadpoolctl` 제한은 나중에 적용되지만 이미 만들어진 스레드의 affinity는 바뀌지 않을 수 있다).
- **지금 못 푸는 이유**: 코드 변경(`pipeline`의 `store` import 지연 또는 패키지 import 구조 조정)이 필요하고, 구조 변경이므로 정확 재현 비교를 동반해야 한다.
- **가능한 접근**: `pipeline.py`의 `store` import를 단계 함수 안으로 옮기거나, API 사용 예처럼 `configure_resources`를 먼저 호출한 뒤 `pipeline`을 import하도록 문서·docstring을 고정한다.

## 실행 출력 디렉터리의 링크·별칭은 검사하지 않음

- **현상**: `run_experiment`는 `output_dir`이 비어 있지 않은지만 검사한다. 빈 디렉터리를 가리키는 심볼릭 링크나 입력 경로와 겹치는 위치를 `output_dir`로 주면 통과한다. 저장소 구축과 준비 스크립트는 `validate_new_paths`로 이런 경우를 거부한다.
- **영향**: 링크를 통해 의도하지 않은 위치(다른 실험 영역)에 결과가 쓰일 수 있다. 비어 있는 경우만 통과하므로 기존 증거가 덮이지는 않는다.
- **지금 못 푸는 이유**: 코드 변경이 필요하다.
- **가능한 접근**: `run_experiment`의 출력 점검을 `validate_new_paths`(링크·입력 별칭 거부)로 바꾸되, 빈 기존 디렉터리 허용 여부를 유지한다.
