#!/usr/bin/env python3
"""
Active Learning Molecular Docking - Installation Test Script
CUDA 호환성 및 전체 시스템 설치 상태를 종합적으로 테스트합니다.
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path

def print_header(text):
    """테스트 섹션 헤더 출력"""
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def print_success(text):
    """성공 메시지 출력"""
    print(f"✓ {text}")

def print_warning(text):
    """경고 메시지 출력"""
    print(f"⚠ {text}")

def print_error(text):
    """오류 메시지 출력"""
    print(f"✗ {text}")

def test_python_environment():
    """Python 환경 테스트"""
    print_header("Python 환경 테스트")
    
    # Python 버전 확인
    py_version = sys.version_info
    print(f"Python 버전: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version.major == 3 and py_version.minor >= 8:
        print_success("Python 버전 호환성 확인")
    else:
        print_error(f"Python 3.8+ 필요 (현재: {py_version.major}.{py_version.minor})")
        return False
    
    # 가상환경 확인
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
        print_success(f"가상환경 활성화됨: {conda_env}")
    else:
        print_warning("가상환경이 활성화되지 않음")
    
    return True

def test_required_packages():
    """필수 패키지 설치 확인"""
    print_header("필수 패키지 설치 확인")
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'scikit-learn',
        'rdkit': 'RDKit',
        'optuna': 'Optuna',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'matplotlib': 'Matplotlib',
        'scipy': 'SciPy',
        'psutil': 'psutil'
    }
    
    missing_packages = []
    installed_versions = {}
    
    for package, display_name in required_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            installed_versions[display_name] = version
            print_success(f"{display_name}: {version}")
        except ImportError:
            missing_packages.append(display_name)
            print_error(f"{display_name}: 설치되지 않음")
    
    if missing_packages:
        print_error(f"누락된 패키지: {', '.join(missing_packages)}")
        return False, {}
    
    print_success("모든 필수 패키지 설치 확인됨")
    return True, installed_versions

def test_cuda_compatibility():
    """CUDA 호환성 종합 테스트"""
    print_header("CUDA 호환성 테스트")
    
    try:
        import torch
        print(f"PyTorch 버전: {torch.__version__}")
        
        # CUDA 사용 가능성 확인
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print_success("CUDA 사용 가능")
            print(f"  - CUDA 버전: {torch.version.cuda}")
            print(f"  - cuDNN 버전: {torch.backends.cudnn.version()}")
            print(f"  - GPU 장치 수: {torch.cuda.device_count()}")
            
            # GPU 정보 출력
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                device_props = torch.cuda.get_device_properties(i)
                print(f"  - GPU {i}: {device_name}")
                print(f"    메모리: {device_props.total_memory / 1024**3:.1f} GB")
                print(f"    계산 능력: {device_props.major}.{device_props.minor}")
            
            # CUDA 연산 테스트
            try:
                print("\nCUDA 연산 테스트 중...")
                device = torch.device('cuda:0')
                
                # 간단한 행렬 연산
                x = torch.randn(1000, 1000).to(device)
                y = torch.randn(1000, 1000).to(device)
                z = torch.mm(x, y)
                
                # GPU에서 CPU로 데이터 이동
                z_cpu = z.cpu()
                
                print_success("CUDA 연산 테스트 성공")
                print(f"  - 테스트 행렬 크기: 1000x1000")
                print(f"  - 결과 합계: {z_cpu.sum().item():.2f}")
                
                # 메모리 사용량 확인
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f"  - GPU 메모리 사용량: {memory_allocated:.2f} GB / {memory_reserved:.2f} GB")
                
                torch.cuda.empty_cache()  # 메모리 정리
                return True, True
                
            except Exception as e:
                print_error(f"CUDA 연산 테스트 실패: {e}")
                print_warning("GPU를 감지했으나 연산에 문제가 있습니다")
                print_warning("라이브러리 버전 충돌일 수 있습니다")
                return True, False
                
        else:
            print_warning("CUDA 사용 불가능 - CPU 모드로 실행")
            print("  가능한 원인:")
            print("  - CUDA가 설치되지 않음")
            print("  - GPU 드라이버 문제")
            print("  - PyTorch CPU 버전 설치됨")
            return False, False
            
    except Exception as e:
        print_error(f"PyTorch/CUDA 테스트 중 오류 발생: {e}")
        return False, False

def test_molecular_libraries():
    """분자 모델링 라이브러리 테스트"""
    print_header("분자 모델링 라이브러리 테스트")
    
    try:
        # RDKit 테스트
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdFingerprintGenerator
        
        # 간단한 분자 생성 및 처리
        smiles = "CCO"  # 에탄올
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            print_success(f"RDKit 분자 처리 테스트 성공")
            print(f"  - 테스트 분자: {smiles} (에탄올)")
            print(f"  - 분자량: {mw:.2f}")
            
            # 핑거프린트 생성 테스트
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
            fp = fpgen.GetFingerprint(mol)
            print_success(f"Morgan 핑거프린트 생성 성공 (길이: {len(fp)})")
        else:
            print_error("RDKit 분자 생성 실패")
            return False
            
    except ImportError as e:
        print_error(f"RDKit 불가능: {e}")
        return False
    except Exception as e:
        print_error(f"RDKit 테스트 오류: {e}")
        return False
    
    try:
        # ODDT 테스트 (선택적)
        import oddt
        print_success(f"ODDT 가능: {oddt.__version__}")
    except ImportError:
        print_warning("ODDT 설치되지 않음 (선택적 패키지)")
    except Exception as e:
        print_warning(f"ODDT 테스트 경고: {e}")
    
    return True

def test_docking_tools():
    """도킹 도구 테스트"""
    print_header("분자 도킹 도구 테스트")
    
    # qvina2.1 바이너리 확인
    qvina_path = Path("bin/qvina2.1")
    if qvina_path.exists():
        if qvina_path.is_file() and os.access(qvina_path, os.X_OK):
            print_success("qVina2.1 바이너리 확인됨")
            try:
                # 버전 확인 시도
                result = subprocess.run([str(qvina_path), "--version"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"  - 버전 정보: {result.stdout.strip()}")
                else:
                    print_warning("qVina2.1 버전 정보 확인 실패")
            except subprocess.TimeoutExpired:
                print_warning("qVina2.1 버전 확인 시간 초과")
            except Exception as e:
                print_warning(f"qVina2.1 테스트 오류: {e}")
        else:
            print_error("qVina2.1 실행 권한 없음")
            print(f"다음 명령으로 권한 설정: chmod +x {qvina_path}")
            return False
    else:
        print_error("qVina2.1 바이너리 없음")
        print("수동 설치 필요:")
        print("  wget https://github.com/QVina/qvina/raw/refs/heads/master/bin/qvina2.1 -O bin/qvina2.1")
        print("  chmod +x bin/qvina2.1")
        return False
    
    return True

def test_project_modules():
    """프로젝트 모듈 로딩 테스트"""
    print_header("프로젝트 모듈 테스트")
    
    project_modules = [
        ('unified_model', 'TabM 모델'),
        ('fingerprint', '분자 핑거프린트'),
        ('molecule_filter', '분자 필터'),
        ('RMSD', 'RMSD 계산'),
        ('Hbond', '수소결합 분석'),
        ('cycle_monitor', '사이클 모니터'),
        ('result_saver', '결과 저장'),
        ('exceptions', '예외 처리')
    ]
    
    loaded_modules = []
    failed_modules = []
    
    for module_name, display_name in project_modules:
        try:
            module = importlib.import_module(module_name)
            loaded_modules.append(display_name)
            print_success(f"{display_name} ({module_name})")
        except ImportError as e:
            failed_modules.append((display_name, str(e)))
            print_error(f"{display_name}: {e}")
        except Exception as e:
            failed_modules.append((display_name, str(e)))
            print_warning(f"{display_name}: {e}")
    
    if failed_modules:
        print_error(f"로딩 실패 모듈 수: {len(failed_modules)}")
        return False
    
    print_success("모든 프로젝트 모듈 로딩 성공")
    return True

def test_configuration():
    """설정 파일 테스트"""
    print_header("설정 파일 테스트")
    
    config_files = [
        'config.yml',
        'environment.yml',
        'environment-cpu.yml'
    ]
    
    found_configs = []
    for config_file in config_files:
        if Path(config_file).exists():
            found_configs.append(config_file)
            print_success(f"{config_file} 확인됨")
        else:
            print_warning(f"{config_file} 없음")
    
    if not found_configs:
        print_error("설정 파일이 없습니다")
        return False
    
    # config.yml 파싱 테스트
    if 'config.yml' in found_configs:
        try:
            import yaml
            with open('config.yml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print_success("config.yml 파싱 성공")
        except Exception as e:
            print_error(f"config.yml 파싱 실패: {e}")
            return False
    
    return True

def generate_system_report(test_results, package_versions, cuda_detected, cuda_working):
    """시스템 리포트 생성"""
    print_header("시스템 설치 리포트")
    
    print(f"테스트 일시: {__import__('datetime').datetime.now()}")
    print(f"Python 버전: {sys.version}")
    print(f"운영체제: {sys.platform}")
    
    # 테스트 결과 요약
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"\n테스트 결과: {passed_tests}/{total_tests} 통과")
    
    for test_name, result in test_results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
    
    # CUDA 상태
    print(f"\nCUDA 상태:")
    if cuda_detected:
        if cuda_working:
            print("  ✓ GPU 가속 사용 가능")
        else:
            print("  ⚠ GPU 감지되었으나 연산 문제 있음")
    else:
        print("  - CPU 전용 모드")
    
    # 권장사항
    print(f"\n권장사항:")
    if not all(test_results.values()):
        print("  - 실패한 테스트를 먼저 해결하세요")
        if not test_results.get('docking_tools', True):
            print("  - qVina2.1 바이너리 수동 설치 필요")
        if not test_results.get('project_modules', True):
            print("  - 프로젝트 모듈 설치 재확인 필요")
    else:
        print("  - 모든 테스트 통과! 시스템이 준비되었습니다")
        if cuda_detected and cuda_working:
            print("  - GPU 가속으로 최고 성능 활용 가능")
        elif cuda_detected:
            print("  - CUDA 문제 해결 시 GPU 가속 활용 가능")

def main():
    """메인 테스트 실행"""
    print("Active Learning Molecular Docking - Installation Test")
    print(f"테스트 시작: {__import__('datetime').datetime.now()}\n")
    
    # 테스트 실행
    test_results = {}
    package_versions = {}
    cuda_detected = False
    cuda_working = False
    
    # 1. Python 환경
    test_results['python_env'] = test_python_environment()
    
    # 2. 필수 패키지
    pkg_ok, versions = test_required_packages()
    test_results['required_packages'] = pkg_ok
    package_versions = versions
    
    # 3. CUDA 호환성
    cuda_detected, cuda_working = test_cuda_compatibility()
    test_results['cuda_compatibility'] = cuda_working if cuda_detected else True  # CPU 모드도 성공으로 간주
    
    # 4. 분자 라이브러리
    test_results['molecular_libraries'] = test_molecular_libraries()
    
    # 5. 도킹 도구
    test_results['docking_tools'] = test_docking_tools()
    
    # 6. 프로젝트 모듈
    test_results['project_modules'] = test_project_modules()
    
    # 7. 설정 파일
    test_results['configuration'] = test_configuration()
    
    # 8. 시스템 리포트
    generate_system_report(test_results, package_versions, cuda_detected, cuda_working)
    
    # 최종 결과
    if all(test_results.values()):
        print_success("\n모든 설치 테스트 통과!")
        return 0
    else:
        print_error("\n일부 테스트 실패 - 문제를 해결한 후 다시 실행하세요")
        return 1

if __name__ == "__main__":
    sys.exit(main())