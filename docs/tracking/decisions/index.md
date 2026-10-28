# 결정 기록

| 번호 | 결정 | 핵심 결과 |
|---|---|---|
| [0001](0001-official-tabm.md) | 자체 MLP 대신 공식 `tabm==0.0.3` | 버전 1 체크포인트 비호환, MC dropout을 TabM 이름으로 되살릴 수 없음 |
| [0002](0002-no-pose-label-filtering.md) | 기준 자세 기반 라벨 검열 제거 | 성공 점수는 모두 라벨, 버전 1 결과와 섞을 수 없음 |
| [0003](0003-exact-reproducibility-gate.md) | 허용 오차 없는 정확 재현을 구조 변경 통과 조건으로 | 결정론적 연산·ID 기반 시드 필수, 방법 변경은 차이 기록 |
| [0004](0004-library-store.md) | 10^8 규모 디스크 기반 라이브러리 저장소 | 저장소 설정과 실행 설정 일치 필수, ID 정렬 유지 필수 |
| [0005](0005-uv-locked-installer.md) | uv 관리형 Python 3.11.13 + 잠금 파일 설치 | 설치 대상 Linux x86_64로 한정, 잠금 파일 두 개 동시 갱신 |
| [0006](0006-hardcoded-resource-policy.md) | CPU 9개·GPU 0/1 한계를 코드에 고정 | 상한 변경은 코드 변경 + 사용자 승인 |
| [0007](0007-isolated-docking-workers.md) | 화합물마다 격리된 자식 프로세스로 도킹 | 시간 제한·장애 격리, 워커 수와 무관한 결과 |
| [0008](0008-ensemble-weighting.md) | 모델 평균 가중, 획득 관측 내부 분할로 검증 가중 | 홀드아웃 누출 없음, validation 전략은 비용 두 배 |
