#!/bin/bash

# Active Learning Molecular Docking - 원클릭 서버 설치 스크립트
# 이 스크립트는 새로운 서버에서 프로젝트를 완전히 설치하고 실행 가능한 상태로 만듭니다.

set -e  # 오류 발생 시 즉시 종료

echo "=========================================="
echo "Active Learning Molecular Docking"
echo "원클릭 서버 설치 스크립트"
echo "=========================================="
echo ""

# 진행 상황 표시 함수  
progress() {
    echo "[$(date '+%H:%M:%S')] ✓ $1"
}

error() {
    echo "[$(date '+%H:%M:%S')] ✗ 오류: $1"
    exit 1
}

warning() {
    echo "[$(date '+%H:%M:%S')] ⚠ 경고: $1"
}

# CUDA 감지 함수
check_cuda() {
    echo "   CUDA 감지 중..."
    
    # nvidia-smi로 CUDA 확인
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
        if [ ! -z "$CUDA_VERSION" ]; then
            progress "CUDA $CUDA_VERSION 감지됨 - GPU 가속 환경 사용"
            return 0
        fi
    fi
    
    # nvcc로 CUDA 확인
    if command -v nvcc &> /dev/null; then
        NVCC_VERSION=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+")
        if [ ! -z "$NVCC_VERSION" ]; then
            progress "NVCC $NVCC_VERSION 감지됨 - GPU 가속 환경 사용"
            return 0
        fi
    fi
    
    warning "CUDA가 감지되지 않음 - CPU 전용 환경 사용"
    return 1
}

# 1. 시스템 요구사항 확인
echo "1. 시스템 요구사항 확인 중..."

# Python 버전 확인
if ! python3 --version &> /dev/null; then
    warning "Python 3가 설치되어 있지 않습니다. Conda로 Python 환경을 구성합니다."
    SYSTEM_PYTHON_OK=false
else
    # 정확한 숫자 비교로 Python 버전 확인
    PYTHON_VERSION_CHECK=$(python3 -c "
import sys
major, minor = sys.version_info.major, sys.version_info.minor
version_tuple = (major, minor)
required_tuple = (3, 8)
if version_tuple >= required_tuple:
    print('OK')
    print(f'{major}.{minor}')
else:
    print('OUTDATED')
    print(f'{major}.{minor}')
")
    
    PYTHON_STATUS=$(echo "$PYTHON_VERSION_CHECK" | head -n1)
    PYTHON_VERSION=$(echo "$PYTHON_VERSION_CHECK" | tail -n1)
    
    if [[ "$PYTHON_STATUS" == "OK" ]]; then
        progress "시스템 Python $PYTHON_VERSION 확인됨 (3.8+ 요구사항 충족)"
        SYSTEM_PYTHON_OK=true
    else
        warning "시스템 Python $PYTHON_VERSION은 3.8 미만입니다. Conda 환경에서 적절한 버전을 설치합니다."
        SYSTEM_PYTHON_OK=false
    fi
fi

# 2. Conda 설치 확인 및 설치
echo ""
echo "2. Conda 환경 설정 중..."

if ! command -v conda &> /dev/null; then
    echo "   Conda가 설치되어 있지 않습니다. Miniconda 설치 중..."
    
    # 시스템 아키텍처 감지
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    else
        error "지원되지 않는 운영체제입니다: $OSTYPE"
    fi
    
    wget -q $MINICONDA_URL -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    
    # 환경변수 설정
    export PATH="$HOME/miniconda3/bin:$PATH"
    $HOME/miniconda3/bin/conda init bash
    source ~/.bashrc 2>/dev/null || true
    
    progress "Miniconda 설치 완료"
else
    progress "Conda 설치 확인됨"
fi

# 3. 프로젝트 환경 생성
echo ""
echo "3. 프로젝트 환경 생성 중..."

# CUDA 지원 여부 확인
if check_cuda; then
    ENV_FILE="environment.yml"
    echo "   GPU 가속 환경 설정 파일 사용: $ENV_FILE"
else
    ENV_FILE="environment-cpu.yml"
    echo "   CPU 전용 환경 설정 파일 사용: $ENV_FILE"
fi

# 기존 환경이 있으면 제거
if conda env list | grep -q "active-learning-docking"; then
    echo "   기존 환경 제거 중..."
    conda env remove -n active-learning-docking -y
fi

# 새 환경 생성
if [[ "$SYSTEM_PYTHON_OK" == "false" ]]; then
    echo "   환경 생성 중 (시스템 Python이 구버전이므로 conda에서 Python 3.8+ 설치, 몇 분 소요될 수 있습니다)..."
else
    echo "   환경 생성 중 (몇 분 소요될 수 있습니다)..."
fi

conda env create -f $ENV_FILE

if [ $? -eq 0 ]; then
    progress "Conda 환경 'active-learning-docking' 생성 완료"
else
    error "환경 생성 실패. $ENV_FILE 파일을 확인하세요."
fi

# 4. qVina2.1 설치 (AutoDock Vina보다 최대 20배 빠름)
echo ""
echo "4. qVina2.1 설치 중..."

# conda 환경 활성화
eval "$(conda shell.bash hook)"
conda activate active-learning-docking

# 바이너리 디렉토리 생성
mkdir -p bin

# 방법 1: 공식 qVina2.1 바이너리 다운로드
echo "   qVina2.1 바이너리 다운로드 시도..."
if wget -q --timeout=30 "https://github.com/QVina/qvina/raw/refs/heads/master/bin/qvina2.1" -O bin/qvina2.1; then
    chmod +x bin/qvina2.1
    progress "qVina2.1 설치 완료 (AutoDock Vina보다 최대 20배 빠름)"
    VINA_SUCCESS=true
# 방법 2: conda를 통한 qvina 설치
elif conda install -c conda-forge qvina -y &>/dev/null; then
    # conda로 설치된 qvina를 bin 디렉토리에 링크
    if command -v qvina2 &> /dev/null; then
        ln -sf $(which qvina2) bin/qvina2.1
    elif command -v qvina &> /dev/null; then
        ln -sf $(which qvina) bin/qvina2.1
    fi
    progress "qVina conda 패키지 설치 완료"
    VINA_SUCCESS=true
# 방법 3: 백업으로 AutoDock Vina 사용
elif wget -q --timeout=30 "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64" -O bin/vina; then
    chmod +x bin/vina
    ln -sf vina bin/qvina2.1
    progress "백업용 AutoDock Vina v1.2.5 설치 완료"
    VINA_SUCCESS=true
elif command -v vina &> /dev/null; then
    # 시스템에 이미 설치된 vina 사용
    ln -sf $(which vina) bin/qvina2.1
    progress "시스템 AutoDock Vina 사용"
    VINA_SUCCESS=true
else
    echo "   경고: qVina2.1 또는 AutoDock Vina 바이너리를 자동 다운로드할 수 없습니다."
    echo "   수동 설치가 필요할 수 있습니다."
    VINA_SUCCESS=false
fi

# 5. 필수 패키지 설치 확인 및 CUDA 호환성 테스트
echo ""
echo "5. 패키지 설치 확인 및 CUDA 호환성 테스트 중..."

python -c "
import sys

# Conda 환경 내 Python 버전 확인
major, minor = sys.version_info.major, sys.version_info.minor
print(f'✓ Conda 환경 Python 버전: {major}.{minor}')

if (major, minor) >= (3, 8):
    print('✓ Python 3.8+ 요구사항 충족됨')
else:
    print(f'✗ Python 버전 문제: {major}.{minor} < 3.8')
    sys.exit(1)

# 필수 패키지 확인
try:
    import torch
    import rdkit
    import pandas
    import numpy
    import sklearn
    import optuna
    print('✓ 모든 필수 패키지 설치됨')
except ImportError as e:
    print(f'✗ 패키지 오류: {e}')
    sys.exit(1)

# PyTorch CUDA 호환성 테스트
try:
    print(f'✓ PyTorch 버전: {torch.__version__}')
    
    if torch.cuda.is_available():
        print(f'✓ CUDA 사용 가능: {torch.version.cuda}')
        print(f'✓ GPU 장치 수: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'  - GPU {i}: {torch.cuda.get_device_name(i)}')
        
        # 간단한 CUDA 연산 테스트
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = x + y
            print('✓ CUDA 연산 테스트 성공')
        except Exception as e:
            print(f'⚠ CUDA 연산 테스트 실패: {e}')
            print('  CPU 모드로 실행하세요')
    else:
        print('✓ CPU 전용 모드로 설정됨')
        
except Exception as e:
    print(f'✗ PyTorch CUDA 테스트 오류: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    progress "패키지 설치 및 CUDA 호환성 확인 완료"
else
    error "패키지 또는 CUDA 호환성 문제 발생"
fi

# 6. 예제 데이터 확인
echo ""
echo "6. 예제 데이터 확인 중..."

# 기본 config.yml 생성 (항상 생성)
echo "   기본 설정 파일을 생성합니다."

cat > config.yml << 'EOF'
# 빠른 테스트용 기본 설정
input:
  protein_file: "inputs/protein.pdb"
  ligand_file: "inputs/reference.pdb" 
  database_file: "inputs/candidates.csv"

output:
  base_dir: "results"
  log_file: "active_learning.log"

active_learning:
  budget: 100                 # 빠른 테스트를 위한 작은 값
  batch_size: 5
  n_cycles: 3

docking:
  executable_path: "./bin/qvina2.1"
  n_cpu: 2

model:
  type: "tabm"
  training:
    epochs: 20               # 빠른 테스트를 위한 작은 값
    
system:
  logging:
    level: "INFO"
    console_output: true
EOF

progress "기본 설정 파일 생성 완료"

# 예제 데이터 확인
if [[ -f "inputs/protein.pdb" ]] && [[ -f "inputs/reference.pdb" ]] && [[ -f "inputs/candidates.csv" ]]; then
    progress "예제 데이터 확인됨"
else
    echo "   예제 데이터가 없습니다. 사용자가 직접 제공해야 합니다."
fi

# 7. 설치 테스트
echo ""
echo "7. 설치 테스트 중..."

# 간단한 import 테스트
python -c "
try:
    from main import main
    print('✓ 프로젝트 모듈 로드 성공')
except ImportError as e:
    print(f'✗ 모듈 오류: {e}')
    import sys; sys.exit(1)
"

progress "설치 테스트 완료"

# 8. 완료 메시지
echo ""
echo "=========================================="
echo "설치 완료!"
echo "=========================================="
echo ""

if [ "$VINA_SUCCESS" = true ]; then
    echo "✓ 모든 구성 요소가 성공적으로 설치되었습니다."
else
    echo "⚠ 대부분의 구성 요소가 설치되었으나 AutoDock Vina 수동 설치가 필요할 수 있습니다."
fi

echo ""
echo "다음 단계:"
echo "1. 환경 활성화: conda activate active-learning-docking"
echo "2. 예제 실행: ./run_example.sh"
echo "3. 또는 직접 실행:"
echo "   python main.py --config config.yml"
echo ""

if [[ -f "inputs/protein.pdb" ]] && [[ -f "inputs/reference.pdb" ]] && [[ -f "inputs/candidates.csv" ]]; then
    echo "바로 실행할 수 있습니다:"
    echo "conda activate active-learning-docking && python main.py --config config.yml"
else
    echo "실행하기 전에 다음 파일들을 inputs/ 폴더에 준비하세요:"
    echo "- protein.pdb (타겟 단백질)"
    echo "- reference.pdb (참조 리간드)"  
    echo "- candidates.csv (화합물 데이터베이스, 'Catalog ID'와 'smiles' 열 필요)"
fi

echo ""
echo "문제가 발생하면 active_learning.log 파일을 확인하세요."
echo "=========================================="