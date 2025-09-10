#!/usr/bin/env python3
"""
리팩토링된 모듈 구조를 테스트하는 스크립트

새로운 모듈들이 제대로 import되고 기본 기능이 작동하는지 확인합니다.
"""

import sys
import logging
import tempfile
import os
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_imports():
    """모든 새 모듈들이 제대로 import되는지 테스트"""
    print("=" * 60)
    print("Testing Module Imports...")
    print("=" * 60)
    
    modules_to_test = [
        'exceptions',
        'data_loader', 
        'docking_engine',
        'sample_acquisition',
        'memory_manager',
        'molecular_processing',
        'Activedocking'  # 리팩토링된 메인 클래스
    ]
    
    failed_imports = []
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name)
            print(f"✅ {module_name} - Successfully imported")
            
            # 주요 클래스들 확인
            if module_name == 'exceptions':
                assert hasattr(module, 'ActiveLearningError')
                assert hasattr(module, 'DataLoadingError')
                assert hasattr(module, 'DockingError')
                print(f"   - Exception classes found")
                
            elif module_name == 'data_loader':
                assert hasattr(module, 'MolecularDataLoader')
                assert hasattr(module, 'CSVDataLoader')
                print(f"   - Data loader classes found")
                
            elif module_name == 'docking_engine':
                assert hasattr(module, 'DockingEngine')
                assert hasattr(module, 'ConformerGenerator')
                print(f"   - Docking engine classes found")
                
            elif module_name == 'sample_acquisition':
                assert hasattr(module, 'SampleAcquisitionStrategy')
                assert hasattr(module, 'QualityFilter')
                print(f"   - Sample acquisition classes found")
                
            elif module_name == 'memory_manager':
                assert hasattr(module, 'MemoryManager')
                assert hasattr(module, 'MemoryMonitor')
                print(f"   - Memory management classes found")
                
            elif module_name == 'molecular_processing':
                assert hasattr(module, 'MolecularProcessor')
                assert hasattr(module, 'FingerprintComputer')
                print(f"   - Molecular processing classes found")
                
            elif module_name == 'Activedocking':
                assert hasattr(module, 'ActiveLearningWithConformers')
                print(f"   - Main orchestrator class found")
                
        except ImportError as e:
            print(f"❌ {module_name} - Import failed: {str(e)}")
            failed_imports.append(module_name)
        except AssertionError as e:
            print(f"⚠️  {module_name} - Missing expected classes")
            failed_imports.append(module_name)
        except Exception as e:
            print(f"❌ {module_name} - Unexpected error: {str(e)}")
            failed_imports.append(module_name)
    
    print(f"\nImport Summary:")
    print(f"✅ Successful: {len(modules_to_test) - len(failed_imports)}/{len(modules_to_test)}")
    if failed_imports:
        print(f"❌ Failed: {failed_imports}")
        return False
    else:
        print(f"🎉 All modules imported successfully!")
        return True

def test_basic_functionality():
    """기본 기능들이 작동하는지 간단히 테스트"""
    print("\n" + "=" * 60)
    print("Testing Basic Functionality...")
    print("=" * 60)
    
    try:
        # 1. Exception 테스트
        from exceptions import ActiveLearningError, DataLoadingError
        print("✅ Exception classes work")
        
        # 2. Memory Manager 테스트
        from memory_manager import MemoryManager, MemoryMonitor
        mm = MemoryManager()
        monitor = MemoryMonitor()
        memory_info = monitor.get_memory_usage()
        assert 'rss_mb' in memory_info
        print("✅ Memory management works")
        
        # 3. 간단한 분자 처리 테스트
        from molecular_processing import MolecularProcessor
        processor = MolecularProcessor()
        stats = processor.get_processing_statistics()
        assert isinstance(stats, dict)
        print("✅ Molecular processing works")
        
        # 4. 샘플 선택 전략 테스트
        from sample_acquisition import ExplorationExploitationStrategy
        strategy = ExplorationExploitationStrategy()
        ratio = strategy.get_exploration_ratio(0, 10)
        assert 0 <= ratio <= 1
        print("✅ Sample acquisition strategy works")
        
        print(f"\n🎉 Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        return False

def test_class_instantiation():
    """주요 클래스들의 인스턴스 생성 테스트"""
    print("\n" + "=" * 60)  
    print("Testing Class Instantiation...")
    print("=" * 60)
    
    try:
        # 임시 파일들 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 가짜 CSV 파일 생성
            csv_file = temp_path / "test.csv"
            csv_file.write_text("Catalog ID,smiles\nC1,CCO\nC2,CCC\n")
            
            # 가짜 PDB 파일들 생성 (최소한의 내용)
            protein_file = temp_path / "protein.pdb"
            protein_file.write_text("ATOM      1  N   ALA A   1      20.154  16.967  20.000  1.00 20.00           N\n")
            
            ligand_file = temp_path / "ligand.pdb"  
            ligand_file.write_text("ATOM      1  C   LIG A   1      20.000  17.000  20.000  1.00 20.00           C\n")
            
            output_dir = temp_path / "output"
            output_dir.mkdir()
            
            print("✅ Test files created")
            
            # ActiveLearningWithConformers 인스턴스 생성 테스트
            try:
                from Activedocking import ActiveLearningWithConformers
                
                # 최소한의 파라미터로 인스턴스 생성
                al = ActiveLearningWithConformers(
                    csv_file=str(csv_file),
                    protein_file=str(protein_file),
                    ligand_file=str(ligand_file),
                    output_dir=str(output_dir),
                    n_cycles=1,  # 테스트용으로 짧게
                    batch_size=1,
                    budget=10
                )
                
                print("✅ ActiveLearningWithConformers instance created successfully")
                
                # 기본 메서드들 존재 확인
                assert hasattr(al, 'load_and_preprocess_data')
                assert hasattr(al, 'run') 
                assert hasattr(al, 'train_model')
                print("✅ Required methods found")
                
                return True
                
            except Exception as e:
                print(f"❌ ActiveLearningWithConformers instantiation failed: {str(e)}")
                # 스택 트레이스 출력 (디버깅용)
                import traceback
                traceback.print_exc()
                return False
                
    except Exception as e:
        print(f"❌ Class instantiation test failed: {str(e)}")
        return False

def run_all_tests():
    """모든 테스트를 실행"""
    print("🚀 Starting Refactoring Tests...")
    
    tests = [
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality), 
        ("Class Instantiation", test_class_instantiation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        success = test_func()
        results.append((test_name, success))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Refactoring appears successful.")
        print("\n📝 Next steps:")
        print("   1. Run the actual application with sample data")
        print("   2. Compare performance with legacy version")
        print("   3. Run comprehensive integration tests")
        return True
    else:
        print("\n⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)