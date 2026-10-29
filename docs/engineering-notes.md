# 실무 노트

## 실행 환경

**공유 서버에서 시간이 터무니없이 측정됨**
- 증상: `unpackbits` 같은 단순 연산이 수 초 걸리거나 같은 작업 시간이 20배씩 흔들린다.
- 원인: 같은 서버의 다른 Python 작업이 만든 메모리 압박(부하 14 안팎에서 관찰). 할당이 멈춘다.
- 대응: 코드를 "최적화"하기 전에 `free -g`로 MemFree를 확인하고, 여유가 돌아오면 다시 잰다.

**환경마다 테스트 결과·패키지 버전이 다르거나 `install.sh --venv .venv`가 거부됨**
- 원인: 저장소에 성격이 다른 환경이 공존한다. `.venv`는 anaconda Python 3.10을 가리키는 예전 개발 환경(잠금 없음, foundation 백엔드 없음), `.venv-ald`는 `install.sh`가 만든 uv 관리 Python 3.11.13 잠금 환경(ruff·pytest·CLI 포함), `.venv-ald-cu128`은 그 CUDA 빌드다. 설치기는 conda 인터프리터와 소유 표시(`.ald-install-owner`)가 없는 기존 환경을 거부한다.
- 대응: 검증·실험은 `.venv-ald/bin/python`(GPU면 `.venv-ald-cu128`)으로 한다. `source activate_env.sh`는 인자가 없으면 `.venv-ald`를, 그것이 없을 때만 `.venv`를 고르므로 어느 환경이 활성화됐는지 `which python`으로 확인한다.

**기본 설정 실행이 "tabpfn requires the supported local tabpfn package"로 바로 실패**
- 원인: TabPFN이 핵심 의존성이 되기 전에 만든 환경(`.venv-ald`, `.venv-ald-cu128` 등)에는 `tabpfn`이 없다. 기본 모델이 `[tabpfn]`이라 `models`를 지정하지 않은 설정은 preflight에서 멈춘다(예산 사용 전).
- 대응: 설치기를 다시 돌리거나(`bash install.sh`, 소유 환경이면 잠금 버전 재적용) 새 `--venv`를 만든다. TabPFN 도입 작업에서 만든 새 환경은 `.venv-ald-tabpfn`(CPU), `.venv-ald-tabpfn-cu128`이다. 예전 환경으로 TabM을 계속 쓰려면 `model.models: [tabm]`을 명시한다.

**CI 통과와 설치 환경 통과는 다른 증거임**
- GitHub Actions는 Python 3.10·3.12에서 `pyproject.toml` 범위로 `[docking,tuning,dev]`(핵심 의존성 `tabpfn` 포함)를 설치해 `ruff check .`, `pytest -m 'not integration and not gpu'`, replay `reproduce`, 빌드를 돌린다. replay `reproduce`는 기본 모델 TabPFN을 CPU로 돌리므로 CI가 TabPFN v2 체크포인트(약 44 MB)를 매번 받는다. 그 밖의 foundation 백엔드, 실제 TabPFN 가중치 테스트(`-m foundation`), 실제 도킹, GNINA는 CI에서 검증되지 않는다.
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
- 기본 모델이 TabM에서 TabPFN으로 바뀐 커밋 **이전**과 비교할 때: 양쪽 모두 **변경 전 worktree 안의 설정 파일을 그 자리에서** 쓰고(현재 번들 YAML에서는 TabM 전용 값이 지워져 있다), `models`를 지정하지 않은 설정에는 양쪽에 `--set 'model.models=[tabm]'`을 준다. 재채점은 끈다. 바꾸지 않으면 TabM 대 TabPFN 비교가 되어 당연히 실패한다.
- 재채점 실행은 `compare_runs`가 `.sdf`(GNINA 입력·출력)도 포즈 바이트로 비교한다. 매니페스트는 비교 대상이 아니므로 SIF 해시·버전 줄 일치는 따로 본다.

**예전 실행과 열 하나가 다름**
- 재채점 실행에는 `docking_energy, cnn_score, cnn_affinity` 열이 더 있다(재채점이 꺼진 실행에는 없다). 점수 정의 자체가 다르므로(−CNNaffinity, pK) 재채점 없는 실행과 비교 대상이 아니다.
- `acquisition_beta` 열이 추가되기 전 실행과는 그 열만 다르다. 중복 구조의 대표 ID 규칙(사전순 최소 ID)이 바뀌기 전 v2 실행과는 중복 구조가 있는 라이브러리에서 선택이 달라질 수 있다. `artifacts/`의 이전 증거와 번들 fixture는 재생성하지 않았다.

**매니페스트에 `git_dirty: true`**
- 커밋되지 않은 변경이 있는 작업 트리에서 실행했다. 보고용 실험은 커밋 후 실행해야 소스와 결과가 연결된다.

## 대용량 라이브러리

**측정 기준값** (8 워커, 코어 0–7, 합성 fragment 라이브러리)
- `prepare-library`: 입력 200만–235만 행 → 895,315개 수용, 1분 43초–2분 11초, 최대 RSS 1.3–1.9 GB.
- 895k 풀 도킹 모드 사이클의 전체 풀 예측(TabM): L40(물리 GPU 1)에서 `inference_batch_size: 8192`로 약 4초. 6질의 실행 전체 27초, RSS 1.4 GB.
- 선형 외삽: 10^8 저장소 구축 약 1.5시간, 사이클당 풀 예측(TabM) 약 8분. 정렬 재작성 단계가 임시 파일을 무작위로 읽으므로 SSD를 쓴다.
- 이 값보다 크게 느리면 코드 회귀 전에 메모리 압박부터 확인한다.

**TabPFN v2 측정 기준값** (2026-10-29, 커밋 `dce8df0`; 위 TabM 값과 별개)
- 환경: L40(물리 GPU 1, 드라이버 580.82.09), PyTorch 2.8.0+cu128, `tabpfn==9.2.0`, CPU 스레드 4(코어 0–3, Xeon Gold 6226R), `.venv-ald-tabpfn-cu128`.
- 데이터: 합성 저장소 151,432개(코어 28종 × 치환기 101종 × 2자리 조합을 RDKit 정규 SMILES로 중복 제거, Morgan 반경 2·2,048비트), 6 워커 구축 12초. 획득 행은 시드 고정 무작위, 라벨은 비트의 고정 무작위 선형 결합 + 잡음(시간에는 영향 없음). 풀 = 나머지 전부.
- 설정: `FoundationRegressor("tabpfn")`, `n_estimators: 8`, `max_features: 500`(500개 선택), `inference_batch_size: 8192`, 고정 실행 옵션(`memory_saving_mode: True` 등) 그대로. 측정 중 GPU 사용률 100%(GPU 제한).

| 문맥(획득) 행 | 학습(`fit`) | 풀 예측 | 처리량 | GPU 최대 할당 | 프로세스 RSS | 10^8 외삽(사이클당 풀 예측) |
|---|---|---|---|---|---|---|
| 1,000 | 16.0초 | 150,432행 414.8초 | 363행/초 | 2.5 GiB | 2.1 GiB | 약 77시간(3.2일) |
| 10,000 | 38.8초 | 141,432행 2,362초 | 60행/초 | 4.2 GiB | 2.5 GiB | 약 464시간(19일) |

- 예측 시간은 풀 크기에 선형, 문맥 크기에 대략 비례한다(문맥 10배 → 행당 약 6배). 위 TabM 기준값(같은 L40에서 895k행 약 4초, 약 22만 행/초)과 비교하면 문맥 1,000에서도 약 600배 느리다. 10^8 풀을 매 사이클 TabPFN으로 전부 예측하는 것은 현실적이지 않으며, 대형 풀에서는 TabM·트리 백엔드를 쓰거나 풀을 줄여야 한다(설계 변경은 하지 않았다).
- 문맥 10,000 측정 중 몇 분 동안 cu128 설치 검증 테스트가 같은 GPU에서 작은 작업을 돌렸다(영향 1% 미만으로 추정).
- 근거·재실행 스크립트: `results/tabpfn_throughput_20261029/`(`gpu1_batch8192_ctx*.json`, `scripts/`). 저장소는 같은 디렉터리의 `store/`.

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
- TabPFN은 학습 시 획득 관측 전체(`X[:]`)를 한 번 풀어서 특성 분산을 계산한다(문맥 추출 전). 획득 수가 예산 이하라 작지만, 위 기준값(TabM)으로 TabPFN 풀 예측 시간을 추정하지 않는다.

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
- `docking.timeout`은 리간드 준비와 모든 형태의 엔진 실행, 재채점이 켜졌으면 SDF 변환과 GNINA 실행까지 합친 시간이다. 시간 초과 시 작업 프로세스의 POSIX 프로세스 그룹을 SIGKILL하고, 이어서 같은 세션 ID의 남은 프로세스를 모두 SIGKILL한 뒤 실패로 기록한다(예산 소모). 형태 수·exhaustiveness를 올리거나 재채점을 켜면 timeout도 함께 올린다(GNINA는 SIF에서 포즈당 약 3.6초).
- GNINA 버전 확인(`apptainer --version`, 이미지 안 `gnina --version`)도 각각 `docking.timeout` 안에서 끝나야 한다.

**GNINA 공식 실행 파일이 이 서버에서 실행되지 않음(`--version` 종료 코드 127)**
- 원인: v1.3.3 릴리스 자산 `gnina.cuda12.8.static`은 이름과 달리 동적 링크이며 GLIBC 2.35+와 `libcudnn.so.9`를 요구한다. 서버는 Ubuntu 20.04(glibc 2.31), cuDNN 9 없음. CPU 전용으로도 안 된다.
- 대응: 공식 Docker 이미지를 다이제스트로 받아 Apptainer SIF로 만들어 rootless Apptainer로 실행한다. 사설 glibc 묶음이나 소스 빌드는 쓰지 않는다.

**GNINA 이미지 준비 중 루트 디스크가 차오름**
- 원인: 공식 Docker 이미지는 37.5 GB다. 공용 Docker 저장소(`/var/lib/docker`, 루트 파일 시스템)에 받거나, 받아 둔 이미지를 `docker-daemon:`/`docker save`로 SIF 변환하면 데몬이 이미지 전체를 루트 임시 디렉터리에 다시 쓴다(실측: 루트 여유 약 46 GB → 약 10 GB).
- 대응: Docker를 거치지 않고 `apptainer build <sif> docker://gnina/gnina@sha256:9da3dc61…`로 레지스트리에서 직접 만든다. 그 전에 `APPTAINER_CACHEDIR`·`APPTAINER_TMPDIR`를 `/data2` 아래로 둔다(레이어 약 23.6 GB, SIF 약 18.3 GB). 이 서버의 레이어 캐시 `/data2/home/jyryu/apptainer/cache`(약 23 GB)는 SIF를 다시 만들 때만 필요하다.

**Apptainer 비특권 설치 스크립트가 `rpm2cpio`를 찾지 못함**
- 원인: 공식 `tools/install-unprivileged.sh`는 RPM을 풀 때 `rpm2cpio`를 쓰는데 이 서버에 없다.
- 대응: `bsdtar -cf - --format=newc @<rpm>`을 실행하는 작은 `rpm2cpio` 대체 스크립트를 PATH 앞에 두고 설치했다(`-d ubuntu20`).

**시간 초과 뒤 `squashfuse_ll` 프로세스·마운트가 남음**
- 원인: 비특권 Apptainer의 SIF 마운트 도우미 `squashfuse_ll`은 작업 프로세스와 같은 세션이지만 **별도 프로세스 그룹**에서 돈다. `os.killpg`만으로는 죽지 않는다.
- 대응: `docking._run_bounded`가 프로세스 그룹 SIGKILL 뒤 `_kill_session`으로 `/proc`을 훑어 같은 세션 ID의 프로세스를 모두 SIGKILL한다(최대 20회 반복). GNINA는 반드시 작업 프로세스의 자식(같은 세션)으로 띄운다. `start_new_session`으로 따로 띄우면 이 정리에서 빠진다.

**재채점 실행이 시작 직후 20–40초 멈춰 있음**
- 원인: 오라클 생성 시 18 GB SIF의 SHA-256을 매 실행 계산한다(매니페스트 `rescoring.image_sha256`). 예산을 쓰기 전 단계다.
- 대응: 정상 동작이다. 해시 생략 경로를 추가하는 것은 출처 기록 정책 변경이다.

**"GNINA image must report exactly one 'gnina v1.3.3' version line"**
- 원인: SIF가 다른 GNINA 버전이거나, 버전 출력 형식이 달라졌다. 이미지 엔트리포인트가 NVIDIA 배너를 앞에 찍어도 `gnina v`로 시작하는 줄만 센다(`apptainer exec`는 엔트리포인트를 거치지 않아 보통 배너가 없다).
- 대응: 공식 다이제스트로 SIF를 다시 만든다. 다른 버전을 쓰려면 아래 "GNINA 업그레이드" 체크리스트를 따른다.

**재채점 화합물이 "no pose MODEL has the recorded docking energy"로 실패**
- 원인: 재채점 포즈는 `REMARK VINA RESULT:` 에너지가 기록된 도킹 에너지와 같은 MODEL이다. 엔진은 에너지를 고정 자릿수로 찍으므로(Vina 3자리, QVina 1자리) 기록값을 그 자릿수로 반올림해 비교한다. Vina Python API 점수는 전체 정밀도라 그대로 비교하면 맞지 않는다.
- 대응: 엔진 출력 형식이 바뀌었는지 확인한다. 반올림 비교를 없애거나 "첫 MODEL"로 바꾸지 않는다(같은 에너지가 여러 개면 파일 안 첫 MODEL이 규칙이다).

**PDBQT와 SDF 입력의 CNNaffinity가 다름**
- 원인: Meeko PDBQT는 일부 방향족 고리 탄소(예: 프탈라지논 `c1n[nH]c(=O)`의 C)를 비방향족 `C`로 적는다. GNINA(OpenBabel)가 SDF에서는 방향족으로 읽어 CNN 입력이 달라진다(Z942299110: 6.4749 대 6.5975).
- 대응: GNINA 입력은 항상 `rescore_pose.sdf`(PDBQT 좌표 + 원래 SMILES 결합 차수)다. PDBQT를 직접 넣거나 원자 타입을 손으로 고치지 않는다. 변환 결과의 정규 SMILES가 입력과 다르면 그 화합물은 실패한다.

**재채점 실행이 "requires an output_dir without ':' or ','" 또는 수용체 경로 오류로 시작 전에 실패**
- 원인: Apptainer bind 문법은 경로의 `:`·`,`를 표현하지 못한다. `output_dir`은 설정 검증에서, 수용체 경로는 오라클 생성 때 검사해 예산을 쓰기 전에 멈춘다.
- 대응: `output_dir`과 수용체 경로(상위 경로 포함)에 `:`·`,`를 쓰지 않는다.

## 모델 백엔드

**TabFM을 골랐는데 `lcb`와 `greedy` 결과가 같음**
- TabFM 공개 회귀 API는 점 예측만 제공해 spread가 항상 0이다. 단독 사용 시 탐색 항이 사라진다. 앙상블에서는 다른 모델과의 불일치로 spread가 생긴다.

**TabFM 옵션이 가중치 로드 전에 거부됨**
- TabFM 1.0.1 PyTorch 회귀기에는 문맥/KV 캐시가 없다. `cache_context`, `maybe_quantize_kv_cache`, `keep_cache_on_device`는 거부된다. 로컬 체크포인트는 `config.json`과 `model.safetensors`가 있는 디렉터리(또는 `regression/` 하위를 가진 부모)여야 하고, `pytorch_model.bin`은 지원하지 않는다. 계산 dtype은 CPU·CUDA 모두 bfloat16이다.

**foundation 백엔드가 "needs at least one varying fingerprint feature" 또는 "At least two acquired labeled samples" 오류로 실패**
- 원인: 선택된 문맥 행의 지문 비트가 하나도 변하지 않거나 문맥 행이 2개 미만이다. TabPFN은 획득 전체로 고른 비트가 추출된 문맥 안에서 하나도 변하지 않아도 이 오류다.
- 대응: 초기 표본을 늘리거나, 더 다양한 분자를 획득하거나, `max_context_samples`를 늘린다.

**"model.<필드>=... applies only to TabM, which is not in model.models"**
- 원인: 기본 모델이 TabPFN으로 바뀐 뒤, TabM 하이퍼파라미터(`n_blocks, d_block, k, dropout, learning_rate, weight_decay, epochs, patience, tuning_method, n_trials`)를 기본값과 다르게 둔 예전 YAML을 `models` 없이 실행했다. 모델이 바뀐 줄 모르고 다른 모델로 돌지 않게 막는 검사다.
- 대응: TabM을 쓸 거면 `model.models: [tabm]`을 넣고, TabPFN을 쓸 거면 그 필드를 지운다. `batch_size`, `validation_fraction`, `inference_batch_size`는 공용이라 남겨도 된다.

**"tabpfn on resources.device: cpu is limited to learning.budget <= 1000"**
- 원인: TabPFN은 CPU에서 문맥 1000행 초과 학습을 거부한다(`allow_cpu_large_dataset: false` 고정). 예산이 1000을 넘으면 도킹 예산을 다 쓴 뒤에 실패할 수 있어 설정 검증에서 미리 막는다.
- 대응: `resources.device: cuda:0` 또는 `cuda:1`. GPU 0은 공유이므로 `nvidia-smi`로 확인한다. 이 검사를 `.env`/`TABPFN_ALLOW_CPU_LARGE_DATASET`로 우회하지 않는다(실행 중 코드가 고정값으로 덮어쓴다).

**TabPFN이 다운로드를 시도했다는 "TabPFN attempted a weight download" 오류**
- 원인: `tabpfn`의 `download_if_not_exists` 경로는 파일이 없으면 고정되지 않은 출처에서 가중치를 받는다. 어댑터는 해시를 확인한 고정 파일 경로만 넘기고, 실행 중 `tabpfn.model_loading.download_model`을 거부 함수로 바꿔 둔다. 이 오류는 그 경로가 실제로 호출됐다는 뜻이다(파일이 사라졌거나 upstream 로딩 경로가 바뀜).
- 대응: 체크포인트 캐시를 확인하고 preflight부터 다시 실행한다. 거부 함수를 없애거나 `tabpfn`을 올리기 전에 로딩 경로를 다시 확인한다.

**TabPFN 체크포인트 로드가 "different fixed execution options" 또는 고정 해시·revision 오류로 실패**
- 원인: 저장 당시와 코드 상수 `_TABPFN_OPTIONS`/`_TABPFN_SETTINGS`가 다르거나, 체크포인트 참조가 고정 파일(`tabpfn-v2-regressor.ckpt`, 고정 SHA-256·revision)이 아니다.
- 대응: 저장 당시 커밋의 코드로 복원한다. 실행 옵션을 바꾸는 것은 과학적 방법 변경이며 이전 체크포인트와 호환되지 않는다.

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
3. `tabpfn`을 올릴 때: 기존 잠금 패키지 버전이 바뀌지 않는지, Python 3.10·3.12에서 `uv pip compile --python-version`으로 풀리는지, v2 회귀 체크포인트를 계속 읽는지, telemetry·라이선스 인증 경로가 생기지 않았는지, `tabpfn.settings` 필드와 `model_loading.download_model` 이름이 그대로인지(preflight가 확인) 본다. 고정 실행 옵션이 바뀌면 이전 체크포인트를 읽지 못한다.

**GNINA 업그레이드(버전·이미지 교체)**
1. 새 공식 이미지 다이제스트를 정하고 `/data2` 아래 캐시·임시 디렉터리로 SIF를 새 경로에 만든다(기존 SIF를 덮어쓰지 않는다).
2. 새 GNINA의 기본 CNN 앙상블 이름과 순서를 소스(`gninasrc/lib/cnn_torch_scorer.cpp`)에서 확인하고, 명시한 결과와 생략한 결과가 바이트 단위로 같은지 본다.
3. `docking.GNINA_VERSION_PREFIX`, `docking.GNINA_CNN_MODELS`를 고치고, 출력 SDF의 `CNNscore`·`CNNaffinity` 항목 형식(자릿수, 0일 때 생략)이 같은지 확인한다.
4. 같은 입력 2회 실행 바이트 일치, `docking.workers` 1과 2 결과 일치를 `ALD_RUN_DOCKING_TESTS=1 ALD_GNINA_IMAGE=<SIF> ALD_APPTAINER=<apptainer>`로 실제 통합 테스트로 확인한다(skip을 통과로 세지 않는다).
5. 점수 정의가 바뀌는 과학적 방법 변경이다. 결정 기록, README·`docs/reference.md`, 커밋 메시지에 남기고 이전 재채점 결과와 섞지 않는다.
