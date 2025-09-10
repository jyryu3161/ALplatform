"""
기본 시스템 테스트

Active Learning 시스템의 핵심 기능이 정상 작동하는지 확인하는 
최소한의 테스트입니다.
"""

import unittest
import sys
import os
import numpy as np

# 상위 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 현재 디렉토리의 test_framework import
try:
    from test_framework import BaseTestCase, TestDataGenerator
except ImportError:
    # 현재 디렉토리에서 직접 import
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from test_framework import BaseTestCase, TestDataGenerator


class BasicImportTest(BaseTestCase):
    """기본 모듈 import 테스트"""
    
    def test_main_import(self):
        """메인 모듈 import 테스트"""
        from main import main
        self.assertTrue(callable(main))
    
    def test_core_modules_import(self):
        """핵심 모듈들 import 테스트"""
        from Activedocking import ActiveLearningWithConformers
        from unified_model import UnifiedModel
        from exceptions import ActiveLearningError
        
        # 클래스들이 제대로 import되었는지 확인
        self.assertTrue(callable(ActiveLearningWithConformers))
        self.assertTrue(callable(UnifiedModel))
        self.assertTrue(issubclass(ActiveLearningError, Exception))


class BasicModelTest(BaseTestCase):
    """기본 모델 기능 테스트"""
    
    def test_unified_model_creation(self):
        """UnifiedModel 생성 테스트"""
        from unified_model import UnifiedModel
        
        model = UnifiedModel(auto_tune=False)
        self.assertIsNotNone(model)
        self.assertEqual(model.model_type, 'tabm')
        self.assertFalse(model.is_fitted)
    
    def test_model_basic_operations(self):
        """모델 기본 동작 테스트"""
        from unified_model import UnifiedModel
        
        model = UnifiedModel(auto_tune=False, epochs=5)  # 빠른 테스트용
        
        # 테스트 데이터 생성
        X = self.create_test_fingerprints(20, 100)
        y = self.create_test_scores(20)
        
        # 모델 훈련
        model.fit(X, y)
        self.assertTrue(model.is_fitted)
        
        # 예측 테스트
        predictions, uncertainties = model.predict(X[:5])
        self.assertEqual(len(predictions), 5)
        self.assertEqual(len(uncertainties), 5)


class BasicUtilityTest(BaseTestCase):
    """기본 유틸리티 테스트"""
    
    def test_test_data_generation(self):
        """테스트 데이터 생성 확인"""
        # 핑거프린트 데이터
        fingerprints = self.create_test_fingerprints(10, 50)
        self.assertEqual(fingerprints.shape, (10, 50))
        self.assertTrue(np.all((fingerprints >= 0) & (fingerprints <= 1)))
        
        # 도킹 스코어
        scores = self.create_test_scores(10)
        self.assertEqual(len(scores), 10)
        self.assertTrue(np.all(scores < 0))  # 도킹 스코어는 일반적으로 음수
    
    def test_smiles_generation(self):
        """SMILES 생성 테스트"""
        smiles_list = TestDataGenerator.create_test_smiles(5)
        self.assertEqual(len(smiles_list), 5)
        self.assertIn("CCO", smiles_list)  # 에탄올이 포함되어야 함


class BasicExceptionTest(BaseTestCase):
    """기본 예외 처리 테스트"""
    
    def test_exception_hierarchy(self):
        """예외 클래스 계층 구조 테스트"""
        from exceptions import (
            ActiveLearningError,
            DataLoadingError,
            ModelTrainingError,
            SamplingError
        )
        
        # 모든 예외가 ActiveLearningError를 상속하는지 확인
        self.assertTrue(issubclass(DataLoadingError, ActiveLearningError))
        self.assertTrue(issubclass(ModelTrainingError, ActiveLearningError))
        self.assertTrue(issubclass(SamplingError, ActiveLearningError))
    
    def test_exception_creation(self):
        """예외 생성 및 발생 테스트"""
        from exceptions import DataLoadingError
        
        with self.assertRaises(DataLoadingError):
            raise DataLoadingError("Test error message")


if __name__ == '__main__':
    # 상세한 출력으로 테스트 실행
    unittest.main(verbosity=2)