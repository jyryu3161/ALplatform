#!/bin/bash
# TabM Active Learning 프로젝트용 conda 환경 활성화 스크립트

echo "TabM Active Learning 환경을 활성화합니다..."

# Conda 초기화 (필요한 경우)
if [ -f ~/anaconda3/bin/conda ]; then
    source ~/anaconda3/bin/activate
elif [ -f /data1/home/dyoh/anaconda3/bin/conda ]; then
    source /data1/home/dyoh/anaconda3/bin/activate
else
    echo "Conda 설치를 찾을 수 없습니다."
    exit 1
fi

# tabm_env 환경 활성화
source /data1/home/dyoh/anaconda3/bin/activate tabm_env

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