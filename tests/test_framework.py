"""
간소화된 테스트 프레임워크

핵심 테스트 기능만 제공하는 최소한의 테스트 프레임워크입니다.
"""

import unittest
import numpy as np
import tempfile
import os
import shutil
from typing import Tuple


class BaseTestCase(unittest.TestCase):
    """모든 테스트의 기본 클래스"""
    
    def setUp(self):
        """테스트 준비"""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """테스트 정리"""
        super().tearDown()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_fingerprints(self, n_samples: int, n_features: int = 1024) -> np.ndarray:
        """테스트용 핑거프린트 데이터 생성"""
        return np.random.randint(0, 2, size=(n_samples, n_features)).astype(np.float32)
    
    def create_test_scores(self, n_samples: int) -> np.ndarray:
        """테스트용 도킹 스코어 생성"""
        return np.random.normal(-6.0, 2.0, n_samples).astype(np.float32)


class TestDataGenerator:
    """테스트 데이터 생성 유틸리티"""
    
    @staticmethod
    def create_test_smiles(count: int = 10) -> list:
        """간단한 테스트용 SMILES 생성"""
        basic_smiles = ["CCO", "CC(C)O", "c1ccccc1", "CCN", "CCC"]
        return [basic_smiles[i % len(basic_smiles)] for i in range(count)]


# 테스트 실행 시 경고 무시 및 랜덤 시드 설정
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)