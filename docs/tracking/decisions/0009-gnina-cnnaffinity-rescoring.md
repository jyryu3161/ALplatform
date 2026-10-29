# 0009. GNINA CNNaffinity 재채점을 학습 라벨로

## 맥락
사용자가 도킹 포즈를 GNINA CNN으로 다시 채점한 값을 라벨로 쓰는 선택지를 추가하기로 했다(2026-10-29). 근거 논문은 JCIM 2025, 65(10), 4833 "Integrating Machine-Learning-Based Pose Sampling with Established Scoring Functions for Virtual Screening"이다. 이 논문에서 가장 좋았던 단일 조합이 "Vina 포즈 + GNINA 채점"이었고, 그 절차("Poses generated with the Vina docking algorithm were scored directly with the Gnina scoring function (--cnn_scoring rescore)")를 따른다. 논문 본문에는 GNINA 출력 중 무엇으로 순위를 매겼는지가 없으며, CNNaffinity는 사용자가 명시한 선택이다. 사용자는 재채점 대상, 단위, 입력 형식, 실행 장치, 실행 방식, 버전 고정 방식을 아래와 같이 정했다.

실행 방식을 정하게 된 사실:
- GNINA 공식 GitHub 릴리스의 최신 안정판은 v1.3.3(2026-06-29, 태그 커밋 `6fe1ce2bb9c35c8067de9f49bb9169857dfbad70`)이다.
- 릴리스 자산은 `gnina.cuda12.8.static` 하나뿐이다(SHA-256 `3340c1f49cd3c7c84d8699182a1c6af13c7fa2a22448d1204640446106f72172`). 이름과 달리 동적 링크 실행 파일이라 GLIBC 2.35 이상과 `libcudnn.so.9`를 요구한다. 이 서버는 Ubuntu 20.04(glibc 2.31)이고 cuDNN 9가 없어, CPU 전용으로도 실행되지 않는다(`--version`이 종료 코드 127).
- 공식 Docker 이미지 `gnina/gnina:v1.3.3`는 디스크에서 37.5 GB를 차지한다. Docker 저장소는 여러 사용자가 함께 쓰는 루트 파일 시스템(`/var/lib/docker`, 당시 사용률 약 95%)에 있다.
- `docker save`/`docker-daemon:` 방식으로 이미지를 내보내는 동안에는 Docker 데몬이 이미지 전체를 루트의 임시 디렉터리에 다시 쓴다. 실제로 루트 여유 공간이 약 10 GB까지 떨어져 이 방식은 중단했다.

## 결정
- **라벨**: GNINA `CNNaffinity`의 부호를 바꾼 값(−CNNaffinity, pK 단위). 작을수록 좋다는 기존 최소화 규약을 유지하기 위한 부호 반전이다. kcal/mol로 변환하지 않는다.
- **재채점 대상**: QVina/Vina 에너지가 가장 낮은 포즈 하나. 최고 형태의 포즈 파일(`best_poses.pdbqt`와 같은 내용)에서 `REMARK VINA RESULT` 에너지가 기록된 도킹 에너지와 같은 MODEL을 고른다(엔진 출력 자릿수 — Vina 3자리, QVina 1자리 — 로 반올림해 비교, 같은 값이 여럿이면 파일 안 첫 MODEL). 엔진은 에너지 순으로 포즈를 쓰므로 보통 첫 MODEL이다. 다른 포즈는 채점하지 않는다.
- **채점 방식**: score-only + `--cnn_scoring rescore`. 최소화(`--minimize`)나 국소 최적화를 하지 않으며 포즈 좌표는 바뀌지 않는다.
- **입력 형식은 결합 차수 SDF**: 좌표는 최고 포즈 PDBQT MODEL에서, 결합 차수는 원래 SMILES에서 가져온다. 변환 절차는 다음과 같으며 결정론적이다(같은 입력의 두 번 변환이 바이트 단위로 같음, meeko 0.7.1, RDKit 2026.03.6).
  1. 엔진 출력 PDBQT에서 위 규칙으로 고른 `MODEL`부터 그 `ENDMDL` 사이의 줄만 취한다(`MODEL`/`ENDMDL` 줄은 뺀다). 이 블록을 `rescore_pose.pdbqt`에 쓴다.
  2. `meeko.PDBQTMolecule(block, is_dlg=False, skip_typing=True)`로 읽는다.
  3. `meeko.RDKitMolCreate.from_pdbqt_mol(...)`로 RDKit 분자를 만든다. Meeko가 리간드 준비 때 PDBQT에 넣은 `REMARK SMILES`/`REMARK SMILES IDX`/`REMARK H PARENT`를 템플릿으로 쓴다. 무거운 원자와 극성 수소의 좌표는 PDBQT 값을 그대로 쓰고, 비극성 수소는 RDKit가 좌표를 계산해 붙인다.
  4. 결과 분자가 정확히 하나이고 conformer가 정확히 하나여야 한다. `Chem.MolToSmiles(Chem.RemoveHs(mol))`가 원래 SMILES의 정규 SMILES와 같아야 한다. 아니면 실패한다.
  5. `Chem.MolToMolBlock(mol) + "$$$$\n"`로 `rescore_pose.sdf`에 쓴다(SD 데이터 항목 없음).
- **켜는 방법과 설정 키**: `docking.rescoring: gnina`(기본은 끔). 이때 `docking.gnina_image`(SIF 파일 경로)와 `docking.apptainer_executable`(Apptainer 실행 파일 경로)이 필수다.
- **실행 위치**: 기존 격리 도킹 작업 프로세스(0007) 안에서, 도킹이 끝난 뒤 같은 화합물 디렉터리에 대해 실행한다. GNINA 실패나 시간 초과는 그 화합물의 도킹 실패로 기록하고 예산 1을 소모한다. 점수를 지어내지 않는다.
- **실행 방식은 rootless Apptainer + SIF**: `/data2`에 둔 SIF 파일로 실행한다. SIF는 공식 Docker Hub 이미지를 다이제스트로 받아 만든다.
  - 원본: `docker://gnina/gnina@sha256:9da3dc61eef272cecb60798933a9a977d028c47024845fdf1f05a941ce5e25a2`(태그 `v1.3.3`, 이미지 config `sha256:281ab4da39cbb250216449caf42aa3b52d21704b3f4e2fdb855d8c62cb7a035e`, 2026-06-30 빌드). 이미지 빌드는 gnina `master`를 클론하며, 빌드 시점 HEAD가 v1.3.3 태그 커밋 `6fe1ce2`였다.
  - SIF: `/data2/home/jyryu/gnina/gnina-1.3.3.sif`(18,348,314,624바이트, SHA-256 `827cc0cfcdb0e965fac95797a4ef3f3b33805be28f61d26a983299c779abd40a`). `apptainer build <sif> docker://gnina/gnina@sha256:9da3dc61…`로 레지스트리에서 직접 만들었다. 같은 다이제스트에서 다시 만들어도 SIF 해시가 같다는 보장은 없으므로, 코드는 SIF 해시가 아니라 안의 `gnina --version`으로 버전을 확인한다. SIF 해시는 실행 매니페스트에 기록한다.
  - Apptainer: 1.5.5(공식 GitHub 릴리스 `apptainer-1.5.5-1.x86_64.rpm`, SHA-256 `e821c129c8ee49b3994816db3b763d1c6cfc69f3ed043af187e07d10dfe6d2c0`). v1.5.5 태그의 `tools/install-unprivileged.sh -d ubuntu20`로 `/data2/home/jyryu/opt/apptainer/1.5.5`에 비특권 설치했다. 실행 파일은 `/data2/home/jyryu/opt/apptainer/1.5.5/bin/apptainer`이다. 이 서버에는 `rpm2cpio`가 없어 `bsdtar -cf - --format=newc @<rpm>`을 대신 쓰는 작은 셸 스크립트를 PATH에 두고 설치했다.
  - 준비 단계(사용자, 실험 실행 전에 한 번): Apptainer 공식 `tools/install-unprivileged.sh`로 설치한다. SIF는 `apptainer build <sif> docker://gnina/gnina@sha256:9da3dc61eef272cecb60798933a9a977d028c47024845fdf1f05a941ce5e25a2`로 만든다(23.6 GB를 내려받음). 이때 `APPTAINER_CACHEDIR`·`APPTAINER_TMPDIR`를 `/data2` 아래로 둔다. 실행 중에는 아무것도 내려받지 않는다.
- **버전 고정**: 지원 버전 접두어를 코드 상수(`docking.GNINA_VERSION_PREFIX = "gnina v1.3.3 "`)로 둔다. SIF 안 `gnina --version`의 표준 출력에서 `gnina v`로 시작하는 줄이 정확히 하나이고 그 줄이 `gnina v1.3.3 `(뒤 공백 포함)으로 시작해야 한다. 이 SIF의 실제 줄은 `gnina v1.3.3 master:6fe1ce2   Built Jun 30 2026.`(공백 3칸)이며, 줄 전체는 매니페스트 `rescoring.version`에 기록한다. 다르면 도킹 전에, 예산을 쓰기 전에 실패한다. GNINA를 올리려면 코드 변경이 필요하다.
- **CNN 모델 명시**: v1.3.3에서 `--cnn`을 생략하면 쓰는 기본 앙상블 `dense_1_3 dense_1_3_PT_KD_3 crossdock_default2018_KD_4`(`gninasrc/lib/cnn_torch_scorer.cpp`)를 이 순서 그대로 명시해 넘긴다(설정 키 없음). 이렇게 하면 업그레이드로 기본값이 바뀌어도 점수 정의가 조용히 바뀌지 않는다. 앙상블 평균은 float32 누적이라 순서도 고정한다. 명시한 결과와 생략한 결과가 바이트 단위로 같음을 확인했다.
- **CPU 전용·격리**: `--nv`를 주지 않는다. 그 위에 `--env CUDA_VISIBLE_DEVICES=`와 GNINA `--no_gpu`를 함께 준다. CPU 집합은 호출 프로세스의 affinity를 그대로 물려받으며, 개수는 `--cpu N`으로 맞춘다. 시드는 `--seed S`로 고정한다. 그 밖의 격리 옵션은 `--containall --cleanenv --no-home --net --network none`이다. 수용체는 읽기 전용으로, 화합물 디렉터리는 쓰기 가능으로 바인드한다.
- **시간 초과는 세션 전체를 SIGKILL**: Apptainer는 새 세션으로 띄운다. 시간 초과 시 프로세스 그룹을 SIGKILL하는 것만으로는 부족하다. 비특권 Apptainer의 SIF 마운트 프로세스 `squashfuse_ll`이 같은 세션 안의 별도 프로세스 그룹에서 돌기 때문에, `os.killpg`만 하면 이 프로세스가 계속 남는다(실측). 그래서 그 세션 ID를 가진 남은 프로세스까지 모두 SIGKILL한다. 0007의 작업 프로세스 정리(`os.killpg`)도 같은 이유로 세션 단위로 넓혔다(`docking._run_bounded` → `_kill_session`, `/proc` 검사). GNINA는 작업 프로세스의 자식으로 같은 세션에서 돌므로 이 정리에 포함된다.
- **명령 템플릿** (`--pwd /`는 `--no-home`에서 생기는 작업 디렉터리 경고를 없앤다):
  ```bash
  <apptainer_executable> exec --containall --cleanenv --no-home --net --network none --pwd / \
    --env CUDA_VISIBLE_DEVICES= \
    --bind <수용체.pdbqt>:/input/receptor.pdbqt:ro --bind <화합물 디렉터리>:/work <gnina_image.sif> \
    gnina --receptor /input/receptor.pdbqt --ligand /work/<포즈>.sdf \
      --score_only --cnn_scoring rescore \
      --cnn dense_1_3 dense_1_3_PT_KD_3 crossdock_default2018_KD_4 \
      --no_gpu --cpu <N> --seed <S> --out /work/gnina_scored.sdf
  ```
- **값 읽기**: 표준 출력은 소수 5자리(`CNNaffinity: 6.59749`)라 쓰지 않는다. `--out` SDF의 데이터 항목을 읽는다.
  - SDF 입력이면 GNINA가 원래 몰블록 뒤에 `> <minimizedAffinity>`(소수 5자리), `> <CNNscore>`, `> <CNNaffinity>`, `> <CNN_VS>`, `> <CNNaffinity_variance>`(모두 고정 소수 10자리)를 붙인다. 예: `> <CNNaffinity>` 다음 줄 `6.5974888802`.
  - 값은 내부적으로 float32이며, 10자리면 float32를 그대로 복원할 수 있다.
  - 파싱 규칙: 결과 파일에 레코드가 정확히 하나이고, 이름이 `CNNaffinity`인 데이터 항목이 정확히 하나 있어야 하며, 값이 유한 실수여야 한다. GNINA는 값이 정확히 0이면 이 항목을 쓰지 않으므로, 항목이 없으면 실패로 처리한다. `CNNscore`도 함께 기록한다.

## 대안
사용자가 채택하지 않은 것:
- **QVina/Vina 점수와 CNNaffinity의 결합 점수·순위 평균·다목적 선택**: 논문에서 순위 평균에 Vina 점수까지 넣으면 GNINA 단독보다 조금 나빴다(BEDROC 0.38 → 0.35). QVina/Vina는 포즈 생성에만 쓰고 에너지는 기록만 한다.
- **CNNscore를 목표값으로**: 사용자가 CNNaffinity를 골랐다.
- **모든 포즈를 재채점하고 CNNscore/CNNaffinity가 가장 좋은 포즈 선택**: 사용자가 기각했다.
- **pK를 kcal/mol로 변환(−1.364×pK)**: 사용자가 기각했다.
- **GPU 재채점**: 사용자가 도킹과 같은 작업 프로세스 안의 CPU 실행을 골랐다. CPU에서 포즈당 약 3.6초(Apptainer SIF 실행; Docker 실행은 약 1.5초)다.
- **PDBQT 포즈를 그대로 입력**: Meeko가 일부 고리 탄소를 비방향족 `C`로 적어 CNN 입력이 달라진다. 아래 "결과"를 보라.
- **PDBQT의 원자 타입을 손으로 보정**: 규칙을 따로 만들어 유지해야 하고, 결합 차수를 복원한 SDF가 더 직접적임.

실행 방식 중 채택하지 않은 것:
- **공식 릴리스 바이너리를 그대로 실행**: 이 서버의 glibc 2.31에서 실행되지 않음.
- **공식 릴리스 바이너리 + 사설 glibc 2.35·cuDNN 9 라이브러리 묶음**: 비공식 실행 환경 조립이라 깨지기 쉽고 검증 부담이 큼.
- **v1.3.3 소스 빌드**: CUDA·libtorch·OpenBabel 의존성 빌드 비용이 큼.
- **이 서버에서 바로 도는 이전 버전 고정**: "최신 안정판 고정" 요구와 맞지 않음.
- **Docker 런타임(`docker run`)**: 동작과 결정론은 확인했다. 그러나 37.5 GB 이미지를 여유 공간이 적은 공유 루트 디스크에 두어야 하고, docker 그룹 권한과 시간 초과 시 `docker rm -f` 정리가 따로 필요하다. 이미지는 SIF를 만든 뒤 Docker 저장소에서 지웠다.

## 결과
- **이전 결과와 섞지 않음**: GNINA 재채점 실행의 점수는 이전 Vina/QVina 에너지 실행과 단위·정의가 달라 비교하거나 섞을 수 없다.
- **업그레이드 비용**: GNINA 업그레이드는 버전 상수·SIF·모델 이름 확인을 포함한 코드 변경이다.
- **입력 형식 비교 (Docker 실행에서 측정)**: 같은 Vina 최고 포즈를 PDBQT로 넣을 때와 결합 차수 SDF로 넣을 때 결과가 달랐다. 수용체는 `examples/data/receptor.pdbqt`이고, 상자 중심은 `inputs/reference.pdb`, 크기 22 Å, exhaustiveness 8, 시드는 `compound_seed(42, ID, SMILES)`다.

  | 화합물 | 입력 | CNNaffinity | CNNscore | minimizedAffinity |
  |---|---|---|---|---|
  | Z942299110 | PDBQT | 6.47485209 | 0.828245997 | −9.00024891 |
  | Z942299110 | SDF | 6.5974888802 | 0.8160941601 | −9.47133 |
  | Z941397906 | PDBQT | 4.90636396 | 0.137299716 | −8.29879093 |
  | Z941397906 | SDF | 4.9063649178 | 0.1372997910 | −8.73315 |

  - Z942299110의 0.12 pK 차이는 Meeko의 방향족 타입 때문이다. Meeko는 프탈라지논의 피리다지논 고리 탄소 두 개(`c1n[nH]c(=O)`의 C)를 PDBQT에 비방향족 `C`로 적는다. GNINA(OpenBabel)는 SDF를 읽을 때 이 탄소를 방향족으로 인식한다. 두 탄소만 `A`로 바꾼 PDBQT의 CNNaffinity는 6.59748983으로 SDF 값과 거의 같다.
  - Vina 상호작용 항 값은 두 형식에서 같았다.
  - 타입이 같아도 두 형식은 float32 몇 ULP(약 1e-6) 차이가 난다. 원자 순서에 따른 누적 순서 차이로 보이며 확인하지 않았다.
  - `minimizedAffinity`의 차이(비율 0.9503)는 회전 가능 결합 수 정규화 차이다. PDBQT는 Meeko `TORSDOF 3`이고 SDF는 OpenBabel이 2개로 센다. 라벨에는 쓰지 않는다.
- **Apptainer 실행 확인 (SDF 입력, `--cpu 4`, CPU 0–3)**:
  - 결과 값이 Docker 실행과 같았다. Z942299110은 CNNaffinity `6.5974888802`, CNNscore `0.8160941601`이고, Z941397906은 `4.9063649178`, `0.1372997910`이다. 결과 SDF의 GNINA 데이터 항목 부분과 표준 출력(`Affinity:` 이후)이 Docker 결과와 바이트 단위로 같았다.
  - 같은 입력으로 두 번 실행한 결과 파일 전체가 바이트 단위로 같았다.
  - 컨테이너 안 네트워크는 호스트와 다른 네임스페이스였다. `/proc/net/dev`에는 `lo`만 있고 라우팅 항목이 없었으며, 외부 TCP 연결이 실패했다(같은 시각 호스트에서는 연결됨). `/sys/class/net`에는 호스트 인터페이스 이름이 보이는데, 이는 sysfs가 호스트 것이기 때문이다.
  - `/dev`에 `nvidia*` 장치가 없었고, `CUDA_VISIBLE_DEVICES`는 비어 있었다.
  - `Cpus_allowed_list: 0-3`과 `nproc=4`로 호출 프로세스의 affinity를 물려받았다.
  - 출력 파일 소유자는 실행 사용자(jyryu)였다.
  - 포즈당 실행 시간은 약 3.6–3.7초다. 빈 컨테이너 시작·종료만은 약 0.16초이고, 나머지는 주로 squashfuse로 SIF 안의 라이브러리를 읽는 시간으로 보인다. Docker에서는 같은 작업이 약 1.5초였다.
  - 표준 출력에는 NVIDIA 배너가 나오지 않았다(Apptainer `exec`는 이미지 엔트리포인트를 거치지 않음). 버전 확인 규칙은 그대로 적용된다.
- **루트 디스크 사건**: 처음에는 이미 받아 둔 Docker 이미지에서 SIF를 만들려고 `docker-daemon:` 원본을 썼다. 그동안 Docker 데몬이 이미지 전체를 루트의 임시 디렉터리에 다시 써서 루트 여유 공간이 약 46 GB에서 약 10 GB까지 떨어졌다. 빌드를 중단하고 Docker 이미지(`sha256:281ab4da39cb…`, 사용하는 컨테이너 없음)를 지운 뒤 레지스트리에서 직접 SIF를 만들었다. 데몬 임시 파일이 정리된 뒤 루트 여유 공간은 약 82 GB로 돌아왔다. 다른 이미지나 컨테이너는 건드리지 않았다.
- **정리한 파일**: SIF를 만들 때 생긴 Apptainer 레이어 캐시(`/data2/home/jyryu/apptainer/cache`, 약 23 GB)와 서버에서 실행되지 않는 공식 릴리스 바이너리(`/data/home/jyryu/opt/gnina/1.3.3`, 2 GB)는 2026-10-30 사용자 결정으로 지웠다. SIF를 다시 만들면 레이어를 다시 내려받는다.
- **라이선스**: SIF는 NVIDIA CUDA 12.6 / Ubuntu 24.04 기반 이미지에서 왔다. 실행할 때마다 NVIDIA Deep Learning Container License 배너를 출력한다.
