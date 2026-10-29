# 0010. 기본 예측 모델을 TabM에서 TabPFN v2로

## 맥락
사용자가 기본 예측 모델을 TabM(0001)에서 사전학습된 표 형식 모델 TabPFN v2로 바꾸기로 했다(2026-10-29). 변경 이유는 사용자 결정으로만 기록되어 있고, 여기에 별도 근거를 덧붙이지 않는다. TabPFN은 획득 관측을 문맥으로 받아 경사 학습 없이 예측하며, 예측 분포 전체를 돌려준다.

정하게 된 사실:
- 공식 `tabpfn` 패키지 9.2.0이 v2 회귀 체크포인트를 읽고, 기존 잠금 패키지(scikit-learn, numpy, torch, scipy, pandas, huggingface-hub 등)의 버전을 하나도 바꾸지 않고 CPU·cu128 잠금 파일에 들어간다. 새로 들어온 잠금 패키지는 `tabpfn`, `skrub`, `lightgbm`, `pydantic-settings`, `python-dotenv`, `requests`, `urllib3`, `charset-normalizer`다. Python 3.10·3.12(CI)와 `constraints-tested.txt` 경로에서도 설치된다.
- v2 회귀 가중치(`Prior-Labs/TabPFN-v2-reg`의 `tabpfn-v2-regressor.ckpt`, 약 44 MB)는 토큰·약관 동의 없이 받을 수 있고, 라이선스는 Prior Labs License 1.1(Apache 2.0 + 출처 표기)이다.
- TabPFN v2의 설계 한계는 문맥 10,000행, 특성 500개다. 지문은 기본 2,048비트다.
- TabPFN 라이브러리는 CPU에서 문맥 1,000행을 넘으면 학습을 거부한다.
- 물리 GPU 1, `resources.deterministic: true`에서 같은 입력을 두 번 실행한 결과가 바이트 단위로 같았다. CPU에서도 같은 설정·시드·환경이면 예측이 정확히 같다.

## 결정
- **기본 모델**: `model.models` 기본값을 `[tabpfn]`으로 바꾼다. TabM은 지우지 않고 `models: [tabm]`으로 계속 고를 수 있다. 자동 튜닝 예제(`config_autotune.yaml`)는 TabM 전용이라 `models: [tabm]`을 명시한다. 앙상블 예제의 TabM 자리는 TabPFN으로 바꾼다(`[tabpfn, random_forest, extra_trees]`, foundation 예제는 `[tabpfn, nori, tabldm, tabfm]`). GPU 재현이 확인되어 `examples/gpu.yaml`도 TabPFN을 쓴다.
- **판과 출처 고정**: TabPFN v2 회귀 체크포인트만 쓴다. 패키지 `tabpfn==9.2.0`(핵심 의존성), Hugging Face revision `02a44705ff7790b50cded404bc59adb94da456a1`, 파일 SHA-256 `2ab5a07d…0736`을 코드에 고정하고, 모델 메타데이터에 패키지 버전·저장소·revision·파일 경로·SHA-256을 남긴다. 복원할 때 다르면 실패한다. 오프라인용 `model_path`도 같은 파일명·해시여야 한다.
- **외부 통신**: 고정 가중치 다운로드(`token=False`)뿐이다. 호스팅 API(`tabpfn-client`)를 쓰지 않는다. 9.2.0에는 telemetry 코드가 없고, 실행 중 라이브러리 자체 다운로드 함수를 거부 함수로 바꿔 고정되지 않은 출처에서 받는 일을 막는다. `.env`/`TABPFN_*`가 계산을 바꾸지 못하게 관련 설정을 실행 중 고정한다.
- **특성 선택**: 매 학습마다 **획득 성공 관측 전체**에서 비트별 분산 상위 `max_features`(기본 500, 1–500)개를 고른다(동점은 비트 번호 순). 학습 행과 예측 행이 같은 열 집합을 쓴다.
- **문맥**: 획득 성공 관측이 `max_context_samples`(기본 10,000, 2–10,000)를 넘으면 시드 비복원 추출로 줄인다. 특성 선택은 추출 **전** 전체 관측으로 한다. 기존 foundation 백엔드(추출 표본 안에서 선택)와 순서가 다르며, 이 차이는 TabPFN에만 적용한다.
- **불확도**: spread는 TabPFN 예측 분포에서 **정확히 계산한 표준편차**(점수 단위)다.
- **설정**: 사용자 파라미터는 `max_features`, `max_context_samples`, `n_estimators`(기본 8), `model_path`뿐이다. 두 상한은 `null`을 받지 않는다. 문맥 처리·캐시 방식, 추론 정밀도 등 재현성·처리량에 영향을 주는 옵션은 코드 상수로 고정하고 `model_info.fixed_upstream_options`에 남긴다.
- **CPU 규칙**: `tabpfn`이 선택되고(앙상블 멤버 포함) `resources.device: cpu`이며 `learning.budget > 1000`이면 설정 검증 오류다. 도킹 예산을 쓴 뒤에야 거부가 드러나는 것을 막는다(사용자 결정).
- **TabM 전용 값 검사**: `tabm`이 선택되지 않았는데 TabM 전용 필드(`n_blocks, d_block, k, dropout, learning_rate, weight_decay, epochs, patience, tuning_method, n_trials`)가 기본값과 다르면 설정 검증 오류다. 연구실 구성원이 모델이 바뀐 줄 모르고 예전 TabM YAML을 돌려 다른 모델로 조용히 실행되는 일을 막는다(사용자 결정). `batch_size`, `validation_fraction`, `inference_batch_size`는 공용이라 검사하지 않는다.
- 앙상블에서 TabPFN은 foundation 멤버로 취급한다(`validation` 전략은 분자 그룹 3개 이상).

## 대안
사용자가 채택하지 않은 것:
- **TabM을 기본으로 유지**: 사용자가 기본 모델 전환을 결정했다. TabM은 선택지로 남긴다.
- **TabM 제거**: 기존 결과 재현, 자동 튜닝, 호환 진입점(`unified_model.py` 등)이 TabM에 기대므로 지우지 않는다.
- **TabPFN 2.5 이상(2.5/2.6/3/3.5)**: 사용자가 v2와 비교한 뒤 고르지 않았다. 특성 한도는 2,000개로 더 크지만 가중치가 비상업 라이선스다.
- **지문 2,048비트를 512비트로 접기(folding)**: 사용자가 "화합물마다 자주 바뀌는 비트가 다르지 않나"를 확인한 뒤 기각하고, 획득 전체 기준 분산 상위 500비트 선택을 택했다. 접기는 서로 다른 비트를 한 열에 섞는다.
- **문맥 표본 안에서 특성 선택(기존 foundation 방식)**: 획득이 10,000을 넘으면 비트는 획득 전체로, 학습은 표본으로 하기로 했다(사용자 선택).
- **분위수로 근사한 spread**(Nori·TabLDM 방식): 사용자가 정확한 분포 표준편차를 골랐다.

## 결과
- **과학적 방법 변경**: 모델이 바뀌므로 `models`를 지정하지 않은 번들 설정(`examples/replay.yaml`, `examples/docking.yaml`, `examples/benchmark.yaml`, `examples/gpu.yaml`, `config_basic.yaml`)과 앙상블·foundation 예제의 질의·예측·지표가 이전과 다르다. 이전 번들 결과, `examples/verification.json`, `artifacts/`, 기존 `results/`는 TabM 기준이며 재생성하지 않았다. 새 결과와 프로토콜 구분 없이 섞지 않는다. 변경 전 커밋과 정확 비교할 때는 변경 전 worktree의 설정 파일로 양쪽에 `model.models=[tabm]`을 준다.
- **spread 의미가 다름**: TabPFN spread는 예측 분포 전체의 퍼짐이라 TabM의 "멤버 간 불일치"와 의미·크기가 다르다. 같은 β라도 탐색 정도가 달라진다.
- **번들 YAML 정리**: TabM이 빠진 번들 설정에서 TabM 전용 필드를 지웠다(그대로 두면 새 검사에 걸린다). 공용 필드는 그대로 둔다.
- **설치·CI**: `tabpfn`이 핵심 의존성이 되어 TabPFN 이전에 만든 환경(`.venv-ald` 등)은 다시 설치해야 기본 설정을 실행할 수 있다. CI의 replay `reproduce`가 매번 v2 가중치(약 44 MB)를 받아 CPU에서 실행한다(사용자가 알고 골랐다). `install.sh --test-models`는 TabPFN 실제 가중치 테스트를 필수로 요구한다.
- **처리량**: 10^8 규모에서 TabPFN 풀 예측은 TabM보다 훨씬 느릴 수 있다. 설계는 바꾸지 않고 측정해 기록한다(`docs/engineering-notes.md` 측정 기준값). TabM 처리량 측정값으로 TabPFN을 추정하지 않는다.
- **업그레이드 비용**: `tabpfn` 패키지·체크포인트·고정 실행 옵션을 바꾸는 것은 코드 변경이며, 이전 TabPFN 체크포인트를 읽지 못하고 결과가 달라진다.
