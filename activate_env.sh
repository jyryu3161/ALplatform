#!/bin/bash
# Active Learning Molecular Docking 프로젝트용 conda 환경 활성화 스크립트

echo "Active Learning 환경을 활성화합니다..."

# Conda 초기화 (자동 감지)
if command -v conda &> /dev/null; then
    # conda 명령이 이미 사용 가능한 경우
    eval "$(conda shell.bash hook)"
elif [ -f ~/anaconda3/bin/conda ]; then
    # 홈 디렉토리의 anaconda3
    eval "$(~/anaconda3/bin/conda shell.bash hook)"
elif [ -f ~/miniconda3/bin/conda ]; then
    # 홈 디렉토리의 miniconda3
    eval "$(~/miniconda3/bin/conda shell.bash hook)"
elif [ -f /opt/conda/bin/conda ]; then
    # 시스템 설치 conda
    eval "$(/opt/conda/bin/conda shell.bash hook)"
else
    echo "오류: Conda 설치를 찾을 수 없습니다."
    echo "다음 중 하나를 실행하세요:"
    echo "  - conda 설치 후 PATH에 추가"
    echo "  - setup_server.sh 스크립트 사용"
    exit 1
fi

# active-learning-docking 환경 활성화
conda activate active-learning-docking

# 환경 확인
echo "현재 활성화된 환경: $(conda info --envs | grep '*' | awk '{print $1}')"

# 주요 패키지 가용성 확인
python -c "
try:
    import rdkit
    import oddt
    import torch
    print('✅ 모든 필수 패키지가 정상적으로 로드됩니다.')
    print(f'RDKit 버전: {rdkit.__version__}')
    print(f'PyTorch 버전: {torch.__version__}')
except ImportError as e:
    print(f'❌ 패키지 로드 실패: {e}')
    import sys; sys.exit(1)
"

echo "환경 설정이 완료되었습니다. 이제 Python 스크립트를 실행할 수 있습니다."