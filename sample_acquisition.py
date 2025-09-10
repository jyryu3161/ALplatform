"""
능동학습 샘플 선택 모듈

불확실성과 탐색-활용 균형을 기반으로 한 샘플 선택 전략을 제공합니다.
다양한 획득 함수(acquisition functions)를 통해 최적의 후보 분자를 선택합니다.
"""

import logging
import gc
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

from exceptions import PredictionError


class AcquisitionFunction(ABC):
    """획득 함수의 추상 기본 클래스"""
    
    @abstractmethod
    def calculate(self, predictions: np.ndarray, uncertainties: np.ndarray, 
                 exploration_ratio: float) -> np.ndarray:
        """
        획득 점수를 계산합니다.
        
        Args:
            predictions: 예측값 배열
            uncertainties: 불확실성 배열
            exploration_ratio: 탐색 비율 (0-1)
            
        Returns:
            획득 점수 배열
        """
        pass


class UCBAcquisition(AcquisitionFunction):
    """Upper Confidence Bound (UCB) 획득 함수"""
    
    def calculate(self, predictions: np.ndarray, uncertainties: np.ndarray, 
                 exploration_ratio: float) -> np.ndarray:
        """UCB 기반 획득 점수 계산"""
        # 정규화
        norm_predictions = self._normalize_array(predictions)
        norm_uncertainties = self._normalize_array(uncertainties)
        
        # UCB = exploitation + exploration
        acquisition_scores = (1 - exploration_ratio) * norm_predictions + exploration_ratio * norm_uncertainties
        
        return acquisition_scores
    
    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """배열을 0-1 범위로 정규화"""
        if len(arr) <= 1:
            return np.zeros_like(arr)
        
        min_val, max_val = np.min(arr), np.max(arr)
        if max_val > min_val:
            return (arr - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(arr)


class ThompsonSamplingAcquisition(AcquisitionFunction):
    """Thompson Sampling 기반 획득 함수"""
    
    def __init__(self, n_samples: int = 1):
        """
        Args:
            n_samples: 샘플링 횟수
        """
        self.n_samples = n_samples
    
    def calculate(self, predictions: np.ndarray, uncertainties: np.ndarray, 
                 exploration_ratio: float) -> np.ndarray:
        """Thompson Sampling 기반 획득 점수 계산"""
        # 가우시안 분포에서 샘플링
        sampled_values = []
        
        for _ in range(self.n_samples):
            samples = np.random.normal(predictions, uncertainties * exploration_ratio)
            sampled_values.append(samples)
        
        # 여러 샘플의 평균 사용
        acquisition_scores = np.mean(sampled_values, axis=0)
        
        return acquisition_scores


class ExplorationExploitationStrategy:
    """탐색-활용 전략 클래스"""
    
    def __init__(self,
                 initial_exploration_ratio: float = 0.8,
                 final_exploration_ratio: float = 0.2,
                 decay_strategy: str = "linear"):
        """
        Args:
            initial_exploration_ratio: 초기 탐색 비율
            final_exploration_ratio: 최종 탐색 비율  
            decay_strategy: 감소 전략 ("linear", "exponential", "cosine")
        """
        self.initial_exploration_ratio = initial_exploration_ratio
        self.final_exploration_ratio = final_exploration_ratio
        self.decay_strategy = decay_strategy
    
    def get_exploration_ratio(self, current_cycle: int, total_cycles: int) -> float:
        """현재 사이클의 탐색 비율을 계산합니다."""
        if total_cycles <= 1:
            return self.initial_exploration_ratio
        
        progress = current_cycle / max(1, total_cycles - 1)
        progress = min(1.0, max(0.0, progress))
        
        if self.decay_strategy == "linear":
            return self._linear_decay(progress)
        elif self.decay_strategy == "exponential":
            return self._exponential_decay(progress)
        elif self.decay_strategy == "cosine":
            return self._cosine_decay(progress)
        else:
            return self._linear_decay(progress)
    
    def _linear_decay(self, progress: float) -> float:
        """선형 감소"""
        return self.initial_exploration_ratio - (self.initial_exploration_ratio - self.final_exploration_ratio) * progress
    
    def _exponential_decay(self, progress: float, alpha: float = 2.0) -> float:
        """지수적 감소"""
        decay_factor = np.exp(-alpha * progress)
        return self.final_exploration_ratio + (self.initial_exploration_ratio - self.final_exploration_ratio) * decay_factor
    
    def _cosine_decay(self, progress: float) -> float:
        """코사인 감소"""
        return self.final_exploration_ratio + 0.5 * (self.initial_exploration_ratio - self.final_exploration_ratio) * (1 + np.cos(np.pi * progress))


class SampleAcquisitionStrategy:
    """
    능동학습 샘플 선택 전략
    
    모델의 예측과 불확실성을 기반으로 다음에 평가할 샘플을 선택합니다.
    """
    
    def __init__(self,
                 batch_size: int = 10,
                 acquisition_function: str = "ucb",
                 exploration_strategy: Optional[ExplorationExploitationStrategy] = None):
        """
        Args:
            batch_size: 선택할 샘플 수
            acquisition_function: 획득 함수 종류 ("ucb", "thompson")
            exploration_strategy: 탐색-활용 전략
        """
        self.batch_size = batch_size
        
        # 획득 함수 설정
        if acquisition_function == "ucb":
            self.acquisition_function = UCBAcquisition()
        elif acquisition_function == "thompson":
            self.acquisition_function = ThompsonSamplingAcquisition()
        else:
            self.acquisition_function = UCBAcquisition()
        
        # 탐색-활용 전략 설정
        self.exploration_strategy = exploration_strategy or ExplorationExploitationStrategy()
        
        # 예측 결과 저장
        self.predicted_scores: Dict[str, float] = {}
        self.prediction_uncertainty: Dict[str, float] = {}
        self.acquisition_scores: Dict[str, float] = {}
        self.cycle_predictions: Dict[int, Dict[str, Dict[str, float]]] = {}
    
    def select_samples(self,
                      screening_indices: List[int],
                      cids: List[str], 
                      model,
                      fingerprint_getter,
                      current_cycle: int,
                      total_cycles: int,
                      batch_processing_size: int = 5000) -> Tuple[List[int], Dict[str, Dict[str, float]]]:
        """
        새로운 샘플을 선택합니다.
        
        Args:
            screening_indices: 스크리닝 대상 인덱스 리스트
            cids: 화합물 ID 리스트
            model: 훈련된 모델
            fingerprint_getter: 핑거프린트 획득 함수
            current_cycle: 현재 사이클 번호
            total_cycles: 전체 사이클 수
            batch_processing_size: 배치 처리 크기
            
        Returns:
            (선택된 인덱스 리스트, 예측 상세 정보)
        """
        if not screening_indices:
            logging.warning("No screening samples available")
            return [], {}
        
        # 현재 사이클의 탐색 비율 계산
        exploration_ratio = self.exploration_strategy.get_exploration_ratio(current_cycle, total_cycles)
        logging.info(f"Exploration ratio for cycle {current_cycle}: {exploration_ratio:.3f}")
        
        # 배치별로 예측 수행
        all_scores = []
        cycle_predictions = {}
        
        for start_idx in range(0, len(screening_indices), batch_processing_size):
            batch_scores, batch_predictions = self._process_batch(
                screening_indices, cids, model, fingerprint_getter,
                start_idx, batch_processing_size, exploration_ratio
            )
            
            all_scores.extend(batch_scores)
            cycle_predictions.update(batch_predictions)
        
        # 사이클 예측 저장
        self.cycle_predictions[current_cycle] = cycle_predictions
        
        if not all_scores:
            logging.warning("No valid predictions for screening samples")
            return [], {}
        
        # 최적 샘플 선택
        selected_indices = self._select_top_samples(all_scores)
        
        logging.info(f"Selected {len(selected_indices)} samples for evaluation")
        
        return selected_indices, cycle_predictions
    
    def _process_batch(self,
                      screening_indices: List[int],
                      cids: List[str],
                      model,
                      fingerprint_getter,
                      start_idx: int,
                      batch_size: int,
                      exploration_ratio: float) -> Tuple[List[Tuple[int, float]], Dict[str, Dict[str, float]]]:
        """배치별 예측 처리"""
        end_idx = min(start_idx + batch_size, len(screening_indices))
        batch_indices = screening_indices[start_idx:end_idx]
        
        batch_cids = [cids[i] for i in batch_indices]
        X_batch, valid_fp_indices = fingerprint_getter(batch_cids)
        
        if len(X_batch) == 0:
            return [], {}
        
        try:
            # 모델 예측
            mean_predictions, std_predictions = model.predict(X_batch)
            
            # 획득 점수 계산
            acquisition_scores = self.acquisition_function.calculate(
                mean_predictions, std_predictions, exploration_ratio
            )
            
            # 정규화된 점수 계산
            exploit_scores = self.acquisition_function._normalize_array(mean_predictions)
            explore_scores = self.acquisition_function._normalize_array(std_predictions)
            
            # 결과 저장
            batch_valid_indices = [batch_indices[i] for i in valid_fp_indices]
            batch_scores = []
            batch_predictions = {}
            
            for i, (idx, pred_score, std_score, acq_score) in enumerate(zip(
                batch_valid_indices, mean_predictions, std_predictions, acquisition_scores)):
                
                cid = cids[idx]
                
                # 전역 저장
                self.predicted_scores[cid] = float(pred_score)
                self.prediction_uncertainty[cid] = float(std_score)
                self.acquisition_scores[cid] = float(acq_score)
                
                # 배치 결과
                batch_scores.append((idx, acq_score))
                batch_predictions[cid] = {
                    'predicted_score': float(pred_score),
                    'uncertainty': float(std_score),
                    'acquisition_score': float(acq_score),
                    'exploitation_score': float(exploit_scores[i]),
                    'exploration_score': float(explore_scores[i])
                }
            
            return batch_scores, batch_predictions
            
        except Exception as e:
            logging.error(f"Error in batch prediction: {str(e)}")
            return [], {}
        
        finally:
            # 메모리 정리
            if 'X_batch' in locals():
                del X_batch
            gc.collect()
    
    def _select_top_samples(self, all_scores: List[Tuple[int, float]]) -> List[int]:
        """획득 점수를 기준으로 최적 샘플을 선택합니다."""
        # 점수 기준 정렬 (높은 점수가 우선)
        sorted_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
        
        # 배치 크기의 2배 선택 (후속 필터링을 위해)
        n_candidates = min(self.batch_size * 2, len(sorted_scores))
        selected_indices = [idx for idx, _ in sorted_scores[:n_candidates]]
        
        return selected_indices
    
    def get_prediction_summary(self, cids: List[str] = None) -> Dict[str, Any]:
        """예측 결과 요약을 반환합니다."""
        if cids is None:
            scores = list(self.predicted_scores.values())
            uncertainties = list(self.prediction_uncertainty.values())
            acquisitions = list(self.acquisition_scores.values())
        else:
            scores = [self.predicted_scores.get(cid, 0) for cid in cids]
            uncertainties = [self.prediction_uncertainty.get(cid, 0) for cid in cids]
            acquisitions = [self.acquisition_scores.get(cid, 0) for cid in cids]
        
        if not scores:
            return {'total_predictions': 0}
        
        return {
            'total_predictions': len(scores),
            'score_stats': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            },
            'uncertainty_stats': {
                'mean': np.mean(uncertainties),
                'std': np.std(uncertainties),
                'min': np.min(uncertainties),
                'max': np.max(uncertainties)
            },
            'acquisition_stats': {
                'mean': np.mean(acquisitions),
                'std': np.std(acquisitions),
                'min': np.min(acquisitions),
                'max': np.max(acquisitions)
            }
        }
    
    def get_cycle_predictions(self, cycle: int) -> Optional[Dict[str, Dict[str, float]]]:
        """특정 사이클의 예측 결과를 반환합니다."""
        return self.cycle_predictions.get(cycle)
    
    def clear_old_predictions(self, keep_cycles: int = 1):
        """오래된 예측 결과를 정리합니다."""
        if len(self.cycle_predictions) <= keep_cycles:
            return
        
        sorted_cycles = sorted(self.cycle_predictions.keys())
        cycles_to_remove = sorted_cycles[:-keep_cycles]
        
        for cycle in cycles_to_remove:
            del self.cycle_predictions[cycle]
            logging.debug(f"Removed cycle {cycle} predictions from memory")
        
        logging.info(f"Cleaned {len(cycles_to_remove)} old cycle predictions")


class QualityFilter:
    """도킹 결과 품질 필터링 클래스"""
    
    def __init__(self,
                 rmsd_threshold: float = 2.0,
                 hbond_threshold: float = 0.3):
        """
        Args:
            rmsd_threshold: RMSD 임계값
            hbond_threshold: H-bond 유사도 임계값
        """
        self.rmsd_threshold = rmsd_threshold
        self.hbond_threshold = hbond_threshold
    
    def filter_valid_compounds(self,
                              candidate_indices: List[int],
                              cids: List[str],
                              docking_scores: Dict[str, float],
                              interaction_data: Dict[str, Dict[str, Any]]) -> List[Tuple[int, str, float, float, float]]:
        """
        품질 기준을 만족하는 화합물을 필터링합니다.
        
        Args:
            candidate_indices: 후보 인덱스 리스트
            cids: 화합물 ID 리스트
            docking_scores: 도킹 점수 딕셔너리
            interaction_data: 상호작용 데이터 딕셔너리
            
        Returns:
            유효한 화합물 정보 리스트 [(idx, cid, score, rmsd, hbond_similarity), ...]
        """
        valid_compounds = []
        
        for idx in candidate_indices:
            cid = cids[idx]
            
            # 도킹 점수 확인
            if cid not in docking_scores:
                continue
            
            score = docking_scores[cid]
            
            # 상호작용 데이터 확인
            if cid not in interaction_data:
                continue
            
            interaction = interaction_data[cid]
            rmsd = interaction['rmsd']
            hbond_similarity = interaction['hbond_similarity']
            
            # 품질 기준 확인
            if rmsd <= self.rmsd_threshold and hbond_similarity >= self.hbond_threshold:
                valid_compounds.append((idx, cid, score, rmsd, hbond_similarity))
        
        return valid_compounds
    
    def select_best_compounds(self, 
                             valid_compounds: List[Tuple[int, str, float, float, float]],
                             n_select: int) -> List[int]:
        """최적 화합물을 선택합니다."""
        if not valid_compounds:
            return []
        
        # 도킹 점수 기준 정렬 (낮은 점수가 더 좋음)
        sorted_compounds = sorted(valid_compounds, key=lambda x: x[2])
        
        # 상위 n_select개 선택
        n_select = min(n_select, len(sorted_compounds))
        selected_indices = [compound[0] for compound in sorted_compounds[:n_select]]
        
        return selected_indices