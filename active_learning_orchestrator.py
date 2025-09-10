"""
능동학습 오케스트레이터 모듈

전체 능동학습 워크플로우를 조율하는 메인 클래스입니다.
각 전문 모듈들을 통합하여 효율적인 분자 스크리닝을 수행합니다.
"""

import os
import logging
import json
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
from tqdm import tqdm

# 전문 모듈들 import
from data_loader import MolecularDataLoader
from docking_engine import DockingEngine
from sample_acquisition import SampleAcquisitionStrategy, QualityFilter, ExplorationExploitationStrategy
from memory_manager import MemoryManager
from molecular_processing import MolecularProcessor
from unified_model import UnifiedModel
from result_saver import ResultSaverMixin
from cycle_monitor import CyclePerformanceMonitor
from exceptions import ActiveLearningError, DataLoadingError, ModelTrainingError


class ActiveLearningOrchestrator(ResultSaverMixin):
    """
    능동학습 오케스트레이터
    
    전체 능동학습 워크플로우를 관리하고 각 전문 모듈들을 조율합니다.
    """
    
    def __init__(self,
                 csv_file: str,
                 protein_file: str,
                 ligand_file: str,
                 output_dir: str,
                 # 기본 파라미터
                 similarity_threshold: float = 0.2,
                 budget: int = 100000,
                 batch_size: int = 2500,
                 n_cycles: int = 10,
                 conf_num: int = 1,
                 hbond_similarity_threshold: float = 0.5,
                 initial_csv: Optional[str] = None,
                 # 필터링 옵션
                 enable_element_filter: bool = True,
                 enable_deduplication: bool = True,
                 # 능동학습 파라미터
                 initial_exploration_ratio: float = 0.8,
                 final_exploration_ratio: float = 0.2,
                 # 모델 파라미터
                 auto_tune: bool = False,
                 tuning_method: str = 'random',
                 max_threads: int = 8,
                 learning_rate: float = 1e-3,
                 dropout: float = 0.2,
                 epochs: int = 100,
                 model_batch_size: int = 256,
                 mc_dropout_samples: int = 50):
        """
        Args:
            csv_file: 메인 화학 라이브러리 CSV 파일
            protein_file: 단백질 PDB 파일
            ligand_file: 기준 리간드 PDB 파일
            output_dir: 출력 디렉터리
            similarity_threshold: 분자 유사도 임계값
            budget: 총 평가할 화합물 수
            batch_size: 사이클당 선택할 화합물 수
            n_cycles: 능동학습 사이클 수
            conf_num: 생성할 형태 수
            hbond_similarity_threshold: H-bond 유사도 임계값
            initial_csv: 초기 화합물 CSV (선택사항)
            enable_element_filter: 원소 필터링 활성화
            enable_deduplication: 중복제거 활성화
            initial_exploration_ratio: 초기 탐색 비율
            final_exploration_ratio: 최종 탐색 비율
            auto_tune: 하이퍼파라미터 자동 튜닝
            tuning_method: 튜닝 방법 ('random', 'grid')
            max_threads: 최대 스레드 수
            learning_rate: 학습률
            dropout: 드롭아웃 비율
            epochs: 에폭 수
            model_batch_size: 모델 배치 크기
            mc_dropout_samples: 몬테카를로 드롭아웃 샘플 수
        """
        super().__init__()
        
        # 기본 설정
        self.csv_file = csv_file
        self.protein_file = protein_file
        self.ligand_file = ligand_file
        self.output_dir = output_dir
        self.initial_csv = initial_csv
        self.similarity_threshold = similarity_threshold
        self.budget = budget
        self.batch_size = batch_size
        self.n_cycles = n_cycles
        self.conf_num = conf_num
        self.hbond_similarity_threshold = hbond_similarity_threshold
        
        # 포즈 디렉터리 설정
        self.pose_dir = os.path.join(output_dir, 'pose')
        os.makedirs(self.pose_dir, exist_ok=True)
        
        # 전문 모듈들 초기화
        self._initialize_modules(
            enable_element_filter, enable_deduplication,
            initial_exploration_ratio, final_exploration_ratio,
            auto_tune, tuning_method, max_threads,
            learning_rate, dropout, epochs, model_batch_size, mc_dropout_samples
        )
        
        # 데이터 저장소
        self.structure_info: Dict[str, str] = {}
        self.cids: List[str] = []
        self.training_indices: List[int] = []
        self.screening_indices: List[int] = []
        
        # 현재 사이클 추적
        self.current_cycle: int = 0
        
        logging.info("Active Learning Orchestrator initialized successfully")
    
    def _initialize_modules(self, enable_element_filter, enable_deduplication,
                           initial_exploration_ratio, final_exploration_ratio,
                           auto_tune, tuning_method, max_threads,
                           learning_rate, dropout, epochs, model_batch_size, mc_dropout_samples):
        """전문 모듈들을 초기화합니다."""
        
        # 1. 데이터 로더
        self.data_loader = MolecularDataLoader(
            self.ligand_file,
            self.similarity_threshold,
            enable_element_filter,
            enable_deduplication
        )
        
        # 2. 분자 처리기
        self.molecular_processor = MolecularProcessor(
            conf_num=self.conf_num,
            fp_radius=2,
            fp_bits=1024
        )
        
        # 3. 도킹 엔진 (나중에 초기화)
        self.docking_engine = None
        
        # 4. 샘플 선택 전략
        exploration_strategy = ExplorationExploitationStrategy(
            initial_exploration_ratio,
            final_exploration_ratio,
            "linear"
        )
        
        self.sample_acquisition = SampleAcquisitionStrategy(
            batch_size=self.batch_size,
            acquisition_function="ucb",
            exploration_strategy=exploration_strategy
        )
        
        # 5. 품질 필터
        self.quality_filter = QualityFilter(
            rmsd_threshold=2.0,
            hbond_threshold=self.hbond_similarity_threshold
        )
        
        # 6. 메모리 관리자
        self.memory_manager = MemoryManager(
            memory_cleanup_interval=5,
            max_cycle_predictions_history=1,
            force_cleanup_threshold_mb=1000.0
        )
        
        # 7. 사이클 모니터
        self.cycle_monitor = CyclePerformanceMonitor(self.output_dir)
        
        # 8. 모델
        self.model = UnifiedModel(
            auto_tune=auto_tune,
            tuning_method=tuning_method,
            max_threads=max_threads,
            learning_rate=learning_rate,
            dropout=dropout,
            epochs=epochs,
            batch_size=model_batch_size,
            mc_dropout_samples=mc_dropout_samples
        )
        
        logging.info(f"Model initialized: {self.model.model_type}")
    
    def _initialize_docking_engine(self):
        """도킹 엔진을 초기화합니다."""
        if self.docking_engine is None:
            self.docking_engine = DockingEngine(
                self.protein_file,
                self.ligand_file,
                self.conf_num,
                './bin/qvina2.1'
            )
            logging.info("Docking engine initialized")
    
    def load_and_preprocess_data(self) -> bool:
        """데이터 로딩 및 전처리를 수행합니다."""
        try:
            logging.info("Starting data loading and preprocessing...")
            
            # 데이터 로딩
            combined_data, invalid_smiles = self.data_loader.load_and_process(
                self.csv_file, self.initial_csv
            )
            
            if not combined_data:
                raise DataLoadingError("No valid compounds loaded")
            
            # 구조 정보 설정
            self.structure_info = combined_data
            self.cids = list(self.structure_info.keys())
            self.screening_indices = list(range(len(self.cids)))
            self.training_indices = []
            
            # 핑거프린트 사전 계산
            if not self.molecular_processor.precompute_fingerprints(self.structure_info):
                raise DataLoadingError("Failed to precompute fingerprints")
            
            # 메모리 관리자에 캐시 등록
            self.memory_manager.register_fingerprint_cache(
                self.molecular_processor.fingerprint_computer.fingerprint_cache
            )
            self.memory_manager.register_prediction_cache(
                self.sample_acquisition.cycle_predictions
            )
            
            logging.info(f"Data preprocessing completed: {len(self.structure_info)} compounds loaded")
            return True
            
        except Exception as e:
            logging.error(f"Data loading failed: {str(e)}")
            return False
    
    def initialize_training_set(self, n_initial: Optional[int] = None) -> bool:
        """초기 훈련 세트를 생성합니다."""
        if n_initial is None:
            n_initial = self.batch_size * 2
        
        # 도킹 엔진 초기화
        self._initialize_docking_engine()
        
        try:
            # 초기 CSV에서 우선 선택
            initial_from_csv = self._select_from_initial_csv(n_initial)
            
            # 부족한 수만큼 메인 라이브러리에서 추가 선택
            remaining_needed = n_initial - len(initial_from_csv)
            additional_from_library = self._select_from_main_library(remaining_needed, initial_from_csv)
            
            # 훈련 세트 구성
            self.training_indices = initial_from_csv + additional_from_library
            self.screening_indices = list(set(self.screening_indices) - set(self.training_indices))
            
            logging.info(f"Initial training set: {len(self.training_indices)} compounds")
            
            if len(self.training_indices) == 0:
                raise ActiveLearningError("No compounds selected for initial training")
            
            # 초기 도킹 수행
            successful_docking = self._perform_initial_docking()
            
            if successful_docking < 2:
                raise ActiveLearningError(f"Insufficient successful docking results: {successful_docking}")
            
            return True
            
        except Exception as e:
            logging.error(f"Training set initialization failed: {str(e)}")
            return False
    
    def _select_from_initial_csv(self, n_initial: int) -> List[int]:
        """초기 CSV에서 화합물을 선택합니다."""
        initial_candidates = self.data_loader.csv_loader.get_loading_statistics().get('initial', {})
        
        if not hasattr(self.data_loader, 'combined_data') or not initial_candidates:
            return []
        
        available_initial = []
        for cid in self.structure_info:
            if cid in initial_candidates:
                idx = self.cids.index(cid)
                if idx in self.screening_indices:
                    available_initial.append(idx)
        
        if not available_initial:
            return []
        
        n_from_initial = min(len(available_initial), n_initial)
        selected = np.random.choice(available_initial, n_from_initial, replace=False).tolist()
        
        logging.info(f"Selected {n_from_initial} compounds from initial CSV")
        return selected
    
    def _select_from_main_library(self, remaining_needed: int, exclude_indices: List[int]) -> List[int]:
        """메인 라이브러리에서 추가 화합물을 선택합니다."""
        if remaining_needed <= 0:
            return []
        
        remaining_indices = list(set(self.screening_indices) - set(exclude_indices))
        
        if not remaining_indices:
            return []
        
        n_from_library = min(remaining_needed, len(remaining_indices))
        selected = np.random.choice(remaining_indices, n_from_library, replace=False).tolist()
        
        logging.info(f"Selected {n_from_library} additional compounds from main library")
        return selected
    
    def _perform_initial_docking(self) -> int:
        """초기 도킹을 수행합니다."""
        logging.info("Performing initial docking...")
        successful_docking = 0
        
        for idx in tqdm(self.training_indices, desc="Initial docking"):
            cid = self.cids[idx]
            smiles = self.structure_info[cid]
            
            score = self.docking_engine.get_docking_score(cid, smiles, self.pose_dir)
            
            if score is not None:
                successful_docking += 1
        
        # 유효한 포즈 카운트
        valid_initial = len(self.docking_engine.get_valid_compounds(2.0, self.hbond_similarity_threshold))
        
        logging.info(f"Initial docking completed. Successful: {successful_docking}")
        logging.info(f"Valid poses (RMSD ≤ 2.0, HBond ≥ {self.hbond_similarity_threshold}): {valid_initial}")
        
        return successful_docking
    
    def train_model(self) -> Tuple[bool, Optional[Dict[str, float]]]:
        """모델을 훈련합니다."""
        self.memory_manager.log_memory_usage("before model training")
        
        try:
            # 유효한 훈련 데이터 준비
            valid_training_data = self._prepare_training_data()
            
            if len(valid_training_data['y']) < 5:
                logging.warning(f"Insufficient valid samples for training: {len(valid_training_data['y'])}")
                return False, None
            
            # 모델 훈련
            logging.info(f"Training {self.model.model_type} with {len(valid_training_data['y'])} compounds")
            
            self.model.fit(valid_training_data['X'], valid_training_data['y'])
            
            # 성능 평가
            metrics = self.model.evaluate_all_metrics(valid_training_data['X'], valid_training_data['y'])
            
            logging.info(f"Training Performance:")
            logging.info(f"  - R²: {metrics['r2']:.4f}")
            logging.info(f"  - MAE: {metrics['mae']:.4f}")
            logging.info(f"  - RMSE: {metrics['rmse']:.4f}")
            logging.info(f"  - Samples: {metrics['n_samples']}")
            
            # 주기적으로 모델 저장
            if hasattr(self, 'current_cycle') and self.current_cycle % 5 == 0:
                self.save_trained_model(self.current_cycle)
            
            self.memory_manager.log_memory_usage("after model training")
            
            return True, metrics
            
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise ModelTrainingError(f"Model training failed: {str(e)}")
    
    def _prepare_training_data(self) -> Dict[str, np.ndarray]:
        """훈련 데이터를 준비합니다."""
        # 유효한 훈련 인덱스 필터링
        valid_indices = []
        for idx in self.training_indices:
            cid = self.cids[idx]
            if (cid in self.docking_engine.docking_scores and
                cid in self.docking_engine.interaction_data):
                
                interaction = self.docking_engine.interaction_data[cid]
                if (interaction['rmsd'] <= 2.0 and
                    interaction['hbond_similarity'] >= self.hbond_similarity_threshold):
                    valid_indices.append(idx)
        
        # CID 리스트 생성
        cid_list = [self.cids[i] for i in valid_indices]
        
        # 핑거프린트 가져오기
        X, valid_fp_indices = self.molecular_processor.get_fingerprint_batch(cid_list)
        
        # 최종 유효한 인덱스
        final_valid_indices = [valid_indices[i] for i in valid_fp_indices]
        
        # 타겟 값 생성
        y = np.array([self.docking_engine.docking_scores[self.cids[i]] for i in final_valid_indices], dtype=np.float32)
        
        return {'X': X, 'y': y, 'indices': final_valid_indices}
    
    def acquire_new_samples(self) -> int:
        """새로운 샘플을 선택합니다."""
        if not self.screening_indices:
            logging.warning("No screening samples available")
            return 0
        
        self.memory_manager.log_memory_usage("before sample acquisition")
        
        try:
            # 샘플 선택
            selected_indices, cycle_predictions = self.sample_acquisition.select_samples(
                self.screening_indices,
                self.cids,
                self.model,
                self.molecular_processor.get_fingerprint_batch,
                self.current_cycle,
                self.n_cycles,
                batch_processing_size=5000
            )
            
            if not selected_indices:
                logging.warning("No samples selected")
                return 0
            
            # 도킹 평가
            new_valid_samples = self._evaluate_selected_samples(selected_indices, cycle_predictions)
            
            self.memory_manager.log_memory_usage("after sample acquisition")
            
            return new_valid_samples
            
        except Exception as e:
            logging.error(f"Sample acquisition failed: {str(e)}")
            return 0
    
    def _evaluate_selected_samples(self, selected_indices: List[int], 
                                  cycle_predictions: Dict[str, Dict[str, float]]) -> int:
        """선택된 샘플들을 평가합니다."""
        cycle_compounds = []
        candidate_results = []
        failed_docking = 0
        
        # 도킹 수행
        for idx in tqdm(selected_indices, desc="Evaluating new samples"):
            cid = self.cids[idx]
            smiles = self.structure_info[cid]
            cycle_compounds.append(cid)
            
            score = self.docking_engine.get_docking_score(cid, smiles, self.pose_dir)
            
            if score is None:
                failed_docking += 1
                continue
            
            # 상호작용 데이터 확인
            if cid in self.docking_engine.interaction_data:
                interaction = self.docking_engine.interaction_data[cid]
                candidate_results.append((idx, cid, score, interaction['rmsd'], interaction['hbond_similarity']))
        
        # 사이클 결과 저장
        if hasattr(self, 'current_cycle'):
            self.save_cycle_detailed_results_with_predictions(self.current_cycle, cycle_compounds)
        
        # 품질 필터링 및 최종 선택
        valid_compounds = self.quality_filter.filter_valid_compounds(
            [result[0] for result in candidate_results],
            self.cids,
            self.docking_engine.docking_scores,
            self.docking_engine.interaction_data
        )
        
        if not valid_compounds:
            logging.warning("No compounds passed quality filters")
            return 0
        
        # 최적 화합물 선택
        selected_samples = self.quality_filter.select_best_compounds(valid_compounds, self.batch_size)
        
        # 훈련 세트 업데이트
        if selected_samples:
            self.training_indices.extend(selected_samples)
            self.screening_indices = list(set(self.screening_indices) - set(selected_samples))
            
            logging.info(f"  - Total candidates docked: {len(selected_indices)}")
            logging.info(f"  - Passed quality filters: {len(valid_compounds)}")
            logging.info(f"  - Selected for training: {len(selected_samples)}")
        
        if failed_docking > 0:
            logging.info(f"Docking failed for {failed_docking} compounds in this cycle")
        
        return len(selected_samples)
    
    def run_active_learning(self) -> Tuple[List[Tuple[str, float]], Optional[Any]]:
        """능동학습 메인 루프를 실행합니다."""
        self.memory_manager.log_memory_usage("at startup")
        
        try:
            logging.info("Starting Active Learning...")
            
            # 1. 데이터 로딩 및 전처리
            if not self.load_and_preprocess_data():
                raise ActiveLearningError("Failed to load and preprocess data")
            
            # 2. 초기 훈련 세트 생성
            if not self.initialize_training_set():
                raise ActiveLearningError("Failed to initialize training set")
            
            # 초기 결과 저장
            initial_compounds = [self.cids[idx] for idx in self.training_indices 
                               if self.cids[idx] in self.docking_engine.docking_scores]
            self.current_cycle = 0
            self.save_cycle_detailed_results(0, initial_compounds)
            
            # 3. 능동학습 루프
            total_valid_compounds = len(self.docking_engine.get_valid_compounds())
            
            for cycle in range(self.n_cycles):
                cycle_number = cycle + 1
                self.current_cycle = cycle_number
                
                logging.info(f"\nStarting cycle {cycle_number}/{self.n_cycles}")
                
                # 사이클 모니터링 시작
                self.cycle_monitor.start_cycle(cycle_number)
                
                # 모델 훈련
                training_success, metrics = self.train_model()
                if not training_success:
                    logging.warning(f"Cycle {cycle_number}: Model training failed")
                    self.cycle_monitor.end_cycle(0, metrics=None)
                    continue
                
                # 새로운 샘플 선택 및 평가
                logging.info("Selecting and evaluating new samples...")
                new_valid_samples = self.acquire_new_samples()
                
                if new_valid_samples == 0:
                    logging.warning(f"Cycle {cycle_number}: No new valid samples found")
                    self.cycle_monitor.end_cycle(0, metrics=metrics)
                    continue
                
                total_valid_compounds += new_valid_samples
                logging.info(f"Cycle {cycle_number}: Added {new_valid_samples} valid samples")
                logging.info(f"Total valid compounds so far: {total_valid_compounds}")
                
                # 결과 저장
                self.save_cycle_results(cycle_number)
                
                # 사이클 모니터링 종료
                compounds_processed = self.batch_size * 2
                self.cycle_monitor.end_cycle(compounds_processed, metrics=metrics)
                
                # 메모리 정리
                if self.memory_manager.should_cleanup_memory(cycle_number):
                    logging.info(f"Performing memory cleanup at cycle {cycle_number}")
                    active_cids = set([self.cids[i] for i in self.training_indices + self.screening_indices])
                    self.memory_manager.cleanup_memory(cycle_number, force_cleanup=True, active_cids=active_cids)
                elif cycle_number % 2 == 0:
                    self.memory_manager.cleanup_memory(cycle_number, force_cleanup=False)
                
                # 예산 확인
                if len(self.training_indices) >= self.budget:
                    logging.info("Reached budget limit. Stopping.")
                    break
            
            # 4. 최종 처리
            logging.info("\nActive Learning completed")
            
            self.save_trained_model()
            self.save_best_model()
            self.save_final_results()
            self.cycle_monitor.log_final_summary()
            self.save_prediction_analysis()
            
            # 최종 결과 필터링
            filtered_results = []
            for cid, score in self.docking_engine.docking_scores.items():
                if cid in self.docking_engine.interaction_data:
                    interaction = self.docking_engine.interaction_data[cid]
                    if (interaction['rmsd'] <= 2.0 and
                        interaction['hbond_similarity'] >= self.hbond_similarity_threshold):
                        filtered_results.append((cid, score))
            
            return sorted(filtered_results, key=lambda x: x[1]), self.model
            
        except Exception as e:
            logging.error(f"Error during active learning execution: {str(e)}")
            raise
        finally:
            self.memory_manager.log_memory_usage("at completion")
    
    def save_trained_model(self, cycle_number: Optional[int] = None):
        """훈련된 모델을 저장합니다."""
        if not self.model.is_fitted:
            logging.warning("Model is not trained yet, cannot save")
            return
        
        if cycle_number is not None:
            model_dir = os.path.join(self.output_dir, 'models', f'cycle_{cycle_number}')
        else:
            model_dir = os.path.join(self.output_dir, 'models', 'final')
        
        try:
            self.model.save_model(model_dir)
            
            # 추가 정보 저장
            additional_info = {
                'cycle': cycle_number,
                'total_training_samples': len(self.training_indices),
                'valid_training_samples': len([idx for idx in self.training_indices 
                                             if self.cids[idx] in self.docking_engine.docking_scores]),
                'similarity_threshold': self.similarity_threshold,
                'hbond_similarity_threshold': self.hbond_similarity_threshold,
            }
            
            info_file = os.path.join(model_dir, 'training_info.json')
            with open(info_file, 'w') as f:
                json.dump(additional_info, f, indent=2)
            
            logging.info(f"Model saved for cycle {cycle_number if cycle_number else 'final'}")
            
        except Exception as e:
            logging.error(f"Failed to save model: {str(e)}")
    
    def save_best_model(self):
        """최적 모델을 저장합니다."""
        best_model_dir = os.path.join(self.output_dir, 'models', 'best')
        
        try:
            self.model.save_model(best_model_dir)
            
            if hasattr(self.model, 'best_params_') and self.model.best_params_:
                best_info = {
                    'best_parameters': self.model.best_params_,
                    'model_type': self.model.model_type,
                    'auto_tuned': self.model.auto_tune,
                    'tuning_method': self.model.tuning_method,
                    'final_training_size': len(self.training_indices)
                }
                
                best_info_file = os.path.join(best_model_dir, 'best_model_info.json')
                with open(best_info_file, 'w') as f:
                    json.dump(best_info, f, indent=2)
            
            logging.info("Best model saved")
            
        except Exception as e:
            logging.error(f"Failed to save best model: {str(e)}")
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """종합 통계를 반환합니다."""
        stats = {}
        
        # 데이터 로딩 통계
        stats['data_loading'] = self.data_loader.get_statistics()
        
        # 분자 처리 통계
        stats['molecular_processing'] = self.molecular_processor.get_processing_statistics()
        
        # 도킹 통계
        if self.docking_engine:
            stats['docking'] = self.docking_engine.get_statistics()
        
        # 샘플 선택 통계
        stats['sample_acquisition'] = self.sample_acquisition.get_prediction_summary()
        
        # 메모리 통계
        stats['memory_management'] = self.memory_manager.get_memory_statistics()
        
        # 사이클 모니터 통계
        stats['cycle_performance'] = self.cycle_monitor.get_summary()
        
        return stats