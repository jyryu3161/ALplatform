# 0005. uv 관리형 Python과 잠금 파일 기반 설치

## 맥락
6개 백엔드(Python 3.11 이상이 필요한 Google TabFM, Git 커밋 고정 TabLDM, setuptools<81이 필요한 Nori)와 도킹·수용체 준비·튜닝·개발 도구를 한 환경에 같은 버전으로 반복 설치해야 했다. 공유 서버에서 sudo, 셸 프로필 수정, 미리 설치된 Python에 의존하지 않아야 했다.

## 결정
`install.sh`가 uv 0.11.7(SHA-256 검증)로 관리형 Python 3.11.13 가상환경을 만들고, `requirements/install.in`에서 생성한 CPU·cu128 잠금 파일의 정확한 버전을 설치한다. uv는 PyPI 패키지와 고정 Git 의존성을 함께 다루고, 관리형 Python 설치와 PyTorch 백엔드(cpu/cu128) 명시 선택을 제공한다. PyTorch 2.8.0 단일 버전을 두 프로필에 공통으로 쓴다. 환경 디렉터리에 소유 표시를 남겨 다른 체크아웃이나 관리되지 않는 환경을 건드리지 않으며, conda 인터프리터와 system-site 환경은 거부한다. 설치 직후 import와 소형 replay로 검증한다.

## 대안
- **conda 환경 파일**(`environment.yml`, `environment-gpu.yml`): 저장소에 남아 있으나 Python 3.11 + 범위 의존성 `[docking,tuning,dev]`만 설치하며, foundation 백엔드·수용체 준비 도구와 정확한 버전 고정이 없다. 지원 설치 경로가 아니다.
- **pip + `pyproject.toml` 범위 의존성**: 수동 설치 경로로 남겨 두지만 잠금이 없어 정확한 환경 재현 근거가 되지 않는다.

## 결과
- 지원 설치 대상은 Linux x86_64, glibc 2.28 이상, Python 3.11.13으로 좁아진다. 패키지 메타데이터는 Python 3.10 이상을 허용하지만 잠금은 그 조합만 보장한다.
- 의존성을 바꾸면 CPU·cu128 잠금 파일을 모두 다시 생성해야 한다.
- `constraints-tested.txt`는 이전 Python 3.10 검증의 과학 패키지 기록으로만 남는다.
- CI는 잠금 파일이 아닌 범위 의존성으로 Python 3.10·3.12를 검사하므로, CI 통과가 설치기 환경 통과를 보장하지 않는다.
