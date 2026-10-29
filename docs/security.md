# 보안·라이선스 정책

## 위협 모델과 사용 주체

이 소프트웨어는 연구실 내부 연구용으로, 한 사용자가 공유 연구 서버에서 직접 실행하는 로컬 CLI·Python 라이브러리다. 로그인, 계정, 역할, 네트워크 서비스가 없다. 실행 주체는 프로세스를 띄운 OS 사용자 하나이며, 파일 시스템 권한이 곧 접근 권한이다.

보호 대상:
1. **실험 증거의 무결성** — 실행 디렉터리, 포즈 파일, 매니페스트 해시. 덮어쓰거나 섞이면 연구 결과의 근거가 사라진다.
2. **공유 서버의 연산 자원** — 다른 사용자 작업과 함께 쓰는 CPU·GPU.
3. **라이선스 조건이 붙은 사전학습 가중치와 그 출력물** — 특히 Google TabFM, 그리고 출처 표기 조건이 있는 TabPFN v2.
4. **설치 공급망** — 내려받는 실행 파일, 패키지, 가중치, GNINA 컨테이너 이미지(SIF).

## 인증

인증 흐름이 없다. API 키, 토큰, 비밀번호, Hugging Face 토큰을 읽거나 저장하지 않는다. foundation 가중치는 공개 저장소에서 인증 없이(`token=False`) 내려받으며, 외부 호스팅 추론 API(예: Synthefy 호스팅 API, TabPFN `tabpfn-client`)는 호출하지 않는다. TabPFN v2 가중치는 토큰·약관 동의 없이 받을 수 있다(그렇지 않게 바뀌면 사용자에게 보고한다). 비밀값이 필요한 기능을 추가하는 것은 이 정책의 변경이며 사용자 결정이 필요하다.

## 허용·금지 행위

| 주체 | 행위 | 조건 | 허용 |
|---|---|---|---|
| 실행 사용자 | 새 출력 디렉터리에 실행 결과 쓰기 | 디렉터리가 없거나 비어 있음 | 허용 |
| 실행 사용자 | 비어 있지 않은 디렉터리에 실험 실행 | 어떤 경우든 | **금지** — 아무것도 쓰기 전에 `FileExistsError` |
| 도킹 오라클 | 이미 있거나 링크인 화합물 포즈 디렉터리·포즈 파일에 쓰기 | 어떤 경우든 | **금지** — 그 화합물의 작업 프로세스를 띄우기 전(실행 도중)에 `FileExistsError`로 실험 중단 |
| 저장소 구축·준비 스크립트 | 이미 있는 경로, 링크(끊긴 링크 포함), 입력과 같은 실제 경로, 생성 경로끼리 충돌 | 어떤 경우든 | **금지** — 첫 출력 전에 사전 점검으로 실패 |
| 파이프라인 | 체크포인트·수치 배열 로드 | pickle 없이(`allow_pickle=False`, `torch.load(weights_only=True)`)·저장된 SHA-256 일치 | 허용, 해시 불일치면 실패 |
| 파이프라인 | 사전학습 가중치 다운로드 | 고정 revision, 파일 해시 기록 | 허용 |
| 파이프라인 | 외부 호스팅 API 호출 | — | **금지** |
| 도킹 작업 프로세스 | `docking.executable`로 지정한 QVina 실행 | 실행 사용자가 설정에 명시 | 허용, 실행 파일 해시는 매니페스트에 기록 |
| 도킹 작업 프로세스 | `docking.apptainer_executable`로 `docking.gnina_image`(GNINA SIF) 실행 | `rescoring: gnina`를 설정에 명시. 루트 권한 없이, 데몬 없이 작업 프로세스의 자식으로 | 허용. 아래 격리 조건을 모두 지킨다 |
| GNINA 컨테이너 | 네트워크 접근, GPU 사용, 호스트 환경 변수·홈 디렉터리 접근, 수용체(읽기 전용)·화합물 디렉터리 밖 바인드 | — | **금지** (`--net --network none`, `--nv` 없음 + `--env CUDA_VISIBLE_DEVICES=` + GNINA `--no_gpu`, `--containall --cleanenv --no-home`, 호출 환경에서 `APPTAINER_*`/`SINGULARITY_*` 변수 제거) |
| 파이프라인 | TabPFN 라이브러리 자체 가중치 다운로드 | — | **금지** — TabPFN 실행 중 `tabpfn.model_loading.download_model`을 거부 함수로 바꾼다. 해시를 확인한 고정 파일만 로드 |
| 도킹 작업 프로세스 | GPU 사용 | — | **금지** (`CUDA_VISIBLE_DEVICES=""`) |
| 모든 프로세스 | 물리 GPU 0·1 이외 사용, CPU 9개 초과 사용 | — | **금지** (이 서버의 운영 규칙) |
| 모든 프로세스 | 상속받은 `CUDA_VISIBLE_DEVICES`보다 넓은 GPU 노출 | — | **금지** — 교집합만 허용 |

설정 파일과 입력 CSV/PDB는 실행 사용자가 제공한 신뢰 입력으로 취급한다. 다만 CSV 문법·필드 수·SMILES·점수 유한성은 항상 검증한다.

## 공급망

- `install.sh`는 uv 0.11.7 압축 파일을 내려받을 때 내장 SHA-256과 일치해야만 압축을 푼다.
- 패키지는 `requirements/install-{cpu,cu128}.lock`의 정확한 버전만 설치하고, 네이티브 패키지(vina, openmm, gemmi, torch, numpy, scipy, scikit-learn, rdkit)는 바이너리 휠만 허용한다.
- TabLDM은 고정 Git 커밋에서 설치한다.
- `bin/qvina2.1`은 Git이 추적하는 실행 파일(정적 ELF, SHA-256 `bc1f908869181d17b48c9a9b4c2a28cce5264ccf3053e15bce32bb5baf83b43d`)로 sdist에도 포함된다. QVina 통합 테스트가 이 파일을 실행하며, `install.sh --test`/`--test-models`는 그 테스트 통과를 요구한다. 출처·버전·라이선스 기록은 없다. 이 파일을 교체하면 해시가 바뀌어 QVina 실행의 매니페스트 기록도 달라진다.
- foundation 가중치는 고정 Hugging Face revision으로 받고, 모델 메타데이터에 저장소·revision·파일 경로·해시를 기록한다. 복원할 때 해시가 다르면 실패한다.
- **TabPFN v2**: `tabpfn==9.2.0`은 핵심 의존성이다(두 잠금 파일에 고정; 함께 추가된 잠금 패키지는 `skrub`, `lightgbm`, `pydantic-settings`, `python-dotenv`, `requests`, `urllib3`, `charset-normalizer`이며 기존 고정 버전은 바뀌지 않았다). 가중치는 `Prior-Labs/TabPFN-v2-reg` revision `02a44705ff7790b50cded404bc59adb94da456a1`의 `tabpfn-v2-regressor.ckpt` 하나만 받는다(SHA-256 `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`, 약 44 MB). 로컬 `model_path`도 이 파일명·해시가 아니면 preflight에서 실패한다. 다른 판(2.5 이상)은 쓰지 않는다.
  - 외부 통신은 이 고정 다운로드뿐이다. `tabpfn` 9.2.0에는 사용 통계(telemetry) 전송 코드가 없다(패키지 소스에서 확인; 버전을 올리면 다시 확인한다). 라이선스 동의용 브라우저 인증(`tabpfn.browser_auth`)은 라이브러리의 다운로드 함수 안에서만 호출되는데, 그 함수는 TabPFN 실행 중 거부 함수로 바뀐다.
  - `tabpfn.settings`는 import 시 `TABPFN_*` 환경 변수와 작업 디렉터리의 `.env`를 읽는다. 계산에 영향을 주는 `allow_cpu_large_dataset`(false), `max_batched_estimator_rows`(null), `max_batched_estimator_cells`(null)는 TabPFN 실행 동안 고정값으로 덮어쓰고 끝나면 되돌린다. 실행 옵션(`fit_mode` 등)도 코드 상수이며 모델 메타데이터 `fixed_upstream_options`에 남는다.
- **GNINA 이미지**: 설치기가 준비하지 않는다(사용자가 실행 전에 한 번 준비).
  - 원본은 공식 Docker Hub 이미지를 다이제스트로 고정한다: `docker://gnina/gnina@sha256:9da3dc61eef272cecb60798933a9a977d028c47024845fdf1f05a941ce5e25a2`(태그 v1.3.3). 이것을 `apptainer build`로 SIF로 바꾼다. 같은 다이제스트로 다시 만든 SIF가 바이트 단위로 같다는 보장이 없으므로, 실행 시 SIF SHA-256을 계산해 매니페스트 `rescoring.image_sha256`에 기록하고, 검사 조건은 이미지 안 `gnina --version`의 `gnina v1.3.3 ` 버전 줄이다. 이 서버의 SIF는 `/data2/home/jyryu/gnina/gnina-1.3.3.sif`(SHA-256 `827cc0cfcdb0e965fac95797a4ef3f3b33805be28f61d26a983299c779abd40a`, 18.3 GB).
  - Apptainer는 공식 `tools/install-unprivileged.sh`로 루트 권한 없이 설치한다(이 서버: 1.5.5, `/data2/home/jyryu/opt/apptainer/1.5.5/bin/apptainer`). 매니페스트에는 Apptainer 경로와 `apptainer --version` 출력만 남고 실행 파일 해시는 남지 않는다.
  - SIF·Apptainer·캐시(`APPTAINER_CACHEDIR`, `APPTAINER_TMPDIR`)는 `/data2` 등 사용자 영역에 둔다. 공유 루트 디스크의 Docker 저장소에 GNINA 이미지(37.5 GB)를 두지 않는다. SIF, GNINA 실행 파일, TabPFN 가중치는 저장소에 커밋하지 않는다.
- conda 인터프리터와 system-site-packages 환경은 설치기가 거부한다.

## 라이선스와 데이터 반출

- **Google TabFM 기본 가중치**: 비상업·비운영 용도로만 사용한다. 가중치 파일과 그 출력물(예측값)을 재배포하지 않는다. 공유하는 실험 산출물에는 가중치 자체가 아니라 참조와 해시만 남긴다.
- 상업 과제나 운영 서비스에 TabFM을 쓰는 것은 금지다. 그런 용도가 생기면 TabFM을 모델 목록에서 빼거나 별도 라이선스를 확인해야 한다.
- **TabPFN v2 가중치**: Prior Labs License 1.1(Apache 2.0 + 출처 표기 조건). 이 가중치로 얻은 결과를 발표·배포할 때 라이선스의 출처 표기 조건을 따르고 TabPFN v2 논문(Hollmann et al., Nature 2025)을 인용한다. `tabpfn` 패키지 자체는 Apache-2.0이다. TabPFN 2.5 이상(비상업 라이선스)은 쓰지 않는다.
- **GNINA 이미지**: NVIDIA CUDA 기반 공식 이미지라 NVIDIA Deep Learning Container License 고지가 붙는다. 이미지·SIF를 다른 곳에 재배포하기 전에 그 라이선스를 확인한다. GNINA 결과를 보고할 때 GNINA 1.0 논문과 근거 논문(JCIM 2025, 65(10), 4833)을 인용하고 매니페스트의 버전 줄·이미지 다이제스트·CNN 모델을 함께 적는다.
- 나머지 백엔드와 도구는 각자의 라이선스를 따르며, 실험 보고에는 사용한 모델·도구·구조·데이터의 출처와 이 저장소의 정확한 커밋을 함께 인용한다.
- `inputs/protein.pdb`와 `inputs/reference.pdb`의 실험적 출처와 재배포 권리는 확인되지 않았다. 두 파일은 GitHub 원격과 sdist(`MANIFEST.in`)에 포함된다. 이 파일로 얻은 결과는 소프트웨어 동작 확인 이상의 과학적 근거로 쓰지 않는다.
- 실행 산출물 디렉터리 `results/`와 `artifacts/`는 Git에서 제외되어 있다. 이 안의 내용을 커밋하지 않는다.

## 기록 대상

매 실행의 `manifest.json`에 반드시 남는 항목:
- 입력 파일 SHA-256 (CSV, 수용체, 기준 리간드, QVina 실행 파일, 저장소의 `library.json`)
- 설치된 배포판 버전 목록, Git revision과 dirty 여부, 패키지 소스 해시
- CPU affinity, 스레드 설정, 요청 장치와 실제 논리 장치, `CUDA_VISIBLE_DEVICES`, 결정론적 연산 여부, GPU 이름/UUID/드라이버(가능할 때)
- 상태(`completed`/`failed`), 실패 시 오류 문자열, 완료 시 모든 산출물의 SHA-256
- 재채점 실행(`rescoring: gnina`)만: 최상위 `rescoring` — SIF 절대 경로와 SHA-256, Apptainer 절대 경로와 보고 버전, GNINA 버전 줄, CNN 모델 이름 3개, 입력 형식(`sdf`). 오라클 생성 직후(예산 사용 전) 기록한다

모델 메타데이터에는 foundation 체크포인트(TabPFN 포함)의 저장소·revision·파일 경로·SHA-256과 TabPFN 고정 실행 옵션이 남는다.

도킹 화합물마다 요청, 결과, 작업 로그, 준비된 형태, 포즈가 남는다. 재채점 실행은 재채점한 포즈(PDBQT·SDF), GNINA 출력 SDF와 로그도 남는다. 실패한 시도도 기록에서 빠지지 않는다.
