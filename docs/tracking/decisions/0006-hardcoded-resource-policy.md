# 0006. 자원 한계를 설정이 아닌 코드에 고정

## 맥락
이 소프트웨어는 다른 사용자 작업과 함께 쓰는 연구 서버에서 돈다. 이 서버에서는 계산 작업이 CPU 9개 이하, 물리 GPU 0·1만 써야 한다(서버 운영 규칙).

## 결정
`config.validate()`, `resources.configure_resources`, `DockingOracle`, `build_library_store`, `TabMRegressor`·`FoundationRegressor`의 장치 검사, 두 준비 스크립트의 `--cpu` 검사가 `cpu_threads`·workers 1–9, `gpu_ids` ⊆ {0,1}, `device` ∈ {cpu, cuda:0, cuda:1}을 각자 직접 검사한다. 설정 파일로 이 상한을 늘릴 수 없다. 상속된 `CUDA_VISIBLE_DEVICES`는 교집합만 취하고, Linux CPU affinity로 자식 프로세스까지 묶는다. 도킹은 `workers × cpu ≤ cpu_threads`를 강제한다.

## 대안
- **상한을 설정값으로 노출**: 다른 기관에서는 편하지만, 이 서버에서 실수로 규칙을 넘는 실행이 가능해지므로 현재는 채택하지 않음. 이 한계는 서버 사정에 따른 것이며 다른 환경으로 옮길 때 바뀔 수 있다(사용자 확인).

## 결과
- 9개를 넘는 CPU나 GPU 2번 이상이 있는 머신에서도 이 코드는 그 자원을 쓰지 못한다. 상한을 바꾸려면 위 검사 위치 전부와 관련 테스트를 함께 고치는 코드 변경이 필요하며 사용자 승인 대상이다.
- 여러 실험을 동시에 띄우면 각 프로세스가 허용 집합의 앞쪽 CPU부터 잡으므로, 겹치지 않게 하려면 실행 전에 `taskset` 등으로 CPU 집합을 나눠야 한다.
