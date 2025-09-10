"""
메모리 관리 모듈

능동학습 과정에서 발생하는 메모리 사용량을 최적화하고,
불필요한 데이터를 정리하여 시스템 안정성을 향상시킵니다.
"""

import gc
import logging
import psutil
import os
from typing import Dict, Set, Optional, Any, List
from abc import ABC, abstractmethod

from exceptions import MemoryError as CustomMemoryError


class MemoryMonitor:
    """메모리 사용량 모니터링 클래스"""
    
    def __init__(self):
        """메모리 모니터 초기화"""
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        self.memory_history = []
    
    def get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량을 반환합니다."""
        try:
            memory_info = self.process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size (물리 메모리)
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': self.process.memory_percent(),  # 전체 시스템 메모리 대비 퍼센트
            }
        except Exception as e:
            logging.warning(f"Failed to get memory usage: {str(e)}")
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}
    
    def log_memory_usage(self, context: str = ""):
        """메모리 사용량을 로깅합니다."""
        memory_info = self.get_memory_usage()
        
        # 피크 메모리 업데이트
        if memory_info['rss_mb'] > self.peak_memory['rss_mb']:
            self.peak_memory = memory_info.copy()
        
        # 히스토리 저장 (최대 100개 항목)
        self.memory_history.append({
            'context': context,
            'memory': memory_info.copy()
        })
        if len(self.memory_history) > 100:
            self.memory_history.pop(0)
        
        logging.info(f"Memory usage {context}: "
                    f"RSS: {memory_info['rss_mb']:.1f}MB, "
                    f"VMS: {memory_info['vms_mb']:.1f}MB, "
                    f"Process: {memory_info['percent']:.1f}%")
    
    def get_memory_increase(self) -> float:
        """초기 메모리 대비 증가량을 반환합니다."""
        current = self.get_memory_usage()
        return current['rss_mb'] - self.initial_memory['rss_mb']
    
    def get_peak_memory(self) -> Dict[str, float]:
        """피크 메모리 사용량을 반환합니다."""
        return self.peak_memory.copy()
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """메모리 사용 통계를 반환합니다."""
        if not self.memory_history:
            return {}
        
        rss_values = [entry['memory']['rss_mb'] for entry in self.memory_history]
        
        return {
            'initial_memory_mb': self.initial_memory['rss_mb'],
            'peak_memory_mb': self.peak_memory['rss_mb'],
            'current_memory_mb': self.get_memory_usage()['rss_mb'],
            'average_memory_mb': sum(rss_values) / len(rss_values),
            'memory_increase_mb': self.get_memory_increase(),
            'samples_count': len(self.memory_history)
        }


class CacheManager(ABC):
    """캐시 관리를 위한 추상 기본 클래스"""
    
    @abstractmethod
    def cleanup_cache(self, keep_active_items: Optional[Set] = None):
        """캐시를 정리합니다."""
        pass
    
    @abstractmethod
    def get_cache_size(self) -> int:
        """캐시 크기를 반환합니다."""
        pass
    
    @abstractmethod
    def get_cache_statistics(self) -> Dict[str, Any]:
        """캐시 통계를 반환합니다."""
        pass


class FingerprintCacheManager(CacheManager):
    """핑거프린트 캐시 관리자"""
    
    def __init__(self, fingerprint_cache: Dict[str, Any]):
        """
        Args:
            fingerprint_cache: 핑거프린트 캐시 딕셔너리 참조
        """
        self.fingerprint_cache = fingerprint_cache
        self.cleanup_count = 0
    
    def cleanup_cache(self, keep_active_items: Optional[Set[str]] = None):
        """핑거프린트 캐시를 정리합니다."""
        if keep_active_items is None:
            # 전체 캐시 정리
            original_size = len(self.fingerprint_cache)
            self.fingerprint_cache.clear()
            logging.info(f"Cleared entire fingerprint cache ({original_size} items)")
            self.cleanup_count += original_size
            return
        
        # 선택적 정리
        cache_keys_to_remove = []
        for cid in self.fingerprint_cache:
            if cid not in keep_active_items:
                cache_keys_to_remove.append(cid)
        
        for cid in cache_keys_to_remove:
            del self.fingerprint_cache[cid]
        
        if cache_keys_to_remove:
            logging.info(f"Cleaned {len(cache_keys_to_remove)} unused fingerprints from cache")
            self.cleanup_count += len(cache_keys_to_remove)
    
    def get_cache_size(self) -> int:
        """캐시 크기를 반환합니다."""
        return len(self.fingerprint_cache)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """캐시 통계를 반환합니다."""
        return {
            'current_size': self.get_cache_size(),
            'total_cleanup_count': self.cleanup_count
        }


class PredictionCacheManager(CacheManager):
    """예측 결과 캐시 관리자"""
    
    def __init__(self, cycle_predictions: Dict[int, Dict]):
        """
        Args:
            cycle_predictions: 사이클별 예측 결과 딕셔너리 참조
        """
        self.cycle_predictions = cycle_predictions
        self.max_keep_cycles = 1
        self.cleanup_count = 0
    
    def cleanup_cache(self, keep_active_items: Optional[Set] = None):
        """예측 결과 캐시를 정리합니다."""
        if len(self.cycle_predictions) <= self.max_keep_cycles:
            return
        
        sorted_cycles = sorted(self.cycle_predictions.keys())
        cycles_to_remove = sorted_cycles[:-self.max_keep_cycles]
        
        for cycle in cycles_to_remove:
            del self.cycle_predictions[cycle]
            self.cleanup_count += 1
            logging.debug(f"Removed cycle {cycle} predictions from memory")
        
        if cycles_to_remove:
            logging.info(f"Cleaned {len(cycles_to_remove)} old cycle predictions")
    
    def set_max_keep_cycles(self, max_cycles: int):
        """유지할 최대 사이클 수를 설정합니다."""
        self.max_keep_cycles = max_cycles
    
    def get_cache_size(self) -> int:
        """캐시 크기를 반환합니다."""
        return len(self.cycle_predictions)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """캐시 통계를 반환합니다."""
        total_predictions = sum(len(pred) for pred in self.cycle_predictions.values())
        
        return {
            'current_cycles': self.get_cache_size(),
            'max_keep_cycles': self.max_keep_cycles,
            'total_predictions': total_predictions,
            'total_cleanup_count': self.cleanup_count
        }


class DataStructureManager:
    """데이터 구조 메모리 관리자"""
    
    def __init__(self):
        """데이터 구조 관리자 초기화"""
        self.cleanup_operations = []
    
    def cleanup_removed_compounds(self,
                                 removed_cids: Set[str],
                                 *data_structures: Dict[str, Any]):
        """제거된 화합물을 모든 데이터 구조에서 정리합니다."""
        total_cleaned = 0
        
        for data_structure in data_structures:
            if data_structure is None:
                continue
            
            keys_to_remove = [cid for cid in removed_cids if cid in data_structure]
            
            for cid in keys_to_remove:
                del data_structure[cid]
                total_cleaned += 1
        
        if total_cleaned > 0:
            logging.info(f"Cleaned {total_cleaned} items from {len(data_structures)} data structures")
            self.cleanup_operations.append({
                'removed_compounds': len(removed_cids),
                'cleaned_items': total_cleaned,
                'affected_structures': len(data_structures)
            })
    
    def get_cleanup_statistics(self) -> Dict[str, Any]:
        """정리 작업 통계를 반환합니다."""
        if not self.cleanup_operations:
            return {'total_operations': 0}
        
        total_cleaned = sum(op['cleaned_items'] for op in self.cleanup_operations)
        total_compounds = sum(op['removed_compounds'] for op in self.cleanup_operations)
        
        return {
            'total_operations': len(self.cleanup_operations),
            'total_cleaned_items': total_cleaned,
            'total_removed_compounds': total_compounds,
            'average_items_per_operation': total_cleaned / len(self.cleanup_operations)
        }


class MemoryManager:
    """
    통합 메모리 관리자
    
    능동학습 과정에서 발생하는 다양한 메모리 사용을 최적화합니다.
    """
    
    def __init__(self,
                 memory_cleanup_interval: int = 5,
                 max_cycle_predictions_history: int = 1,
                 force_cleanup_threshold_mb: float = 1000.0):
        """
        Args:
            memory_cleanup_interval: 정리 작업 수행 간격 (사이클 단위)
            max_cycle_predictions_history: 유지할 최대 사이클 예측 수
            force_cleanup_threshold_mb: 강제 정리 임계값 (MB)
        """
        self.memory_cleanup_interval = memory_cleanup_interval
        self.max_cycle_predictions_history = max_cycle_predictions_history
        self.force_cleanup_threshold_mb = force_cleanup_threshold_mb
        
        # 컴포넌트 초기화
        self.monitor = MemoryMonitor()
        self.cache_managers: List[CacheManager] = []
        self.data_structure_manager = DataStructureManager()
        
        # 정리 작업 카운터
        self.cleanup_operations = 0
        self.last_cleanup_cycle = 0
    
    def register_fingerprint_cache(self, fingerprint_cache: Dict[str, Any]):
        """핑거프린트 캐시를 등록합니다."""
        cache_manager = FingerprintCacheManager(fingerprint_cache)
        self.cache_managers.append(cache_manager)
        logging.info("Fingerprint cache registered for memory management")
    
    def register_prediction_cache(self, cycle_predictions: Dict[int, Dict]):
        """예측 결과 캐시를 등록합니다."""
        cache_manager = PredictionCacheManager(cycle_predictions)
        cache_manager.set_max_keep_cycles(self.max_cycle_predictions_history)
        self.cache_managers.append(cache_manager)
        logging.info("Prediction cache registered for memory management")
    
    def should_cleanup_memory(self, cycle_number: int) -> bool:
        """메모리 정리가 필요한지 확인합니다."""
        # 주기적 정리
        if cycle_number % self.memory_cleanup_interval == 0:
            return True
        
        # 메모리 사용량 기반 정리
        current_memory = self.monitor.get_memory_usage()
        if current_memory['rss_mb'] > self.force_cleanup_threshold_mb:
            logging.warning(f"Memory usage ({current_memory['rss_mb']:.1f}MB) "
                          f"exceeds threshold ({self.force_cleanup_threshold_mb}MB)")
            return True
        
        return False
    
    def cleanup_memory(self, 
                      cycle_number: int,
                      force_cleanup: bool = False,
                      active_cids: Optional[Set[str]] = None):
        """
        메모리 정리를 수행합니다.
        
        Args:
            cycle_number: 현재 사이클 번호
            force_cleanup: 강제 정리 여부
            active_cids: 활성 화합물 ID 집합
        """
        try:
            self.monitor.log_memory_usage(f"before cleanup (cycle {cycle_number})")
            
            # 캐시 관리자들을 통한 정리
            for cache_manager in self.cache_managers:
                if isinstance(cache_manager, FingerprintCacheManager) and force_cleanup:
                    cache_manager.cleanup_cache(active_cids)
                elif isinstance(cache_manager, PredictionCacheManager):
                    cache_manager.cleanup_cache()
            
            # 가비지 컬렉션 수행
            collected = gc.collect()
            logging.debug(f"Garbage collection freed {collected} objects")
            
            self.monitor.log_memory_usage(f"after cleanup (cycle {cycle_number})")
            
            self.cleanup_operations += 1
            self.last_cleanup_cycle = cycle_number
            
            logging.info(f"Memory cleanup completed for cycle {cycle_number}")
            
        except Exception as e:
            logging.warning(f"Memory cleanup failed: {str(e)}")
            raise CustomMemoryError(f"Memory cleanup failed: {str(e)}")
    
    def cleanup_removed_compounds(self,
                                 removed_cids: Set[str],
                                 *data_structures: Dict[str, Any]):
        """제거된 화합물들을 모든 데이터 구조에서 정리합니다."""
        self.data_structure_manager.cleanup_removed_compounds(removed_cids, *data_structures)
    
    def log_memory_usage(self, context: str = ""):
        """메모리 사용량을 로깅합니다."""
        self.monitor.log_memory_usage(context)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """종합 메모리 통계를 반환합니다."""
        memory_stats = self.monitor.get_memory_statistics()
        
        # 캐시 통계 수집
        cache_stats = {}
        for i, cache_manager in enumerate(self.cache_managers):
            cache_type = type(cache_manager).__name__
            cache_stats[f"{cache_type}_{i}"] = cache_manager.get_cache_statistics()
        
        # 데이터 구조 통계
        data_structure_stats = self.data_structure_manager.get_cleanup_statistics()
        
        return {
            'memory_usage': memory_stats,
            'cache_statistics': cache_stats,
            'data_structure_cleanup': data_structure_stats,
            'cleanup_operations': self.cleanup_operations,
            'last_cleanup_cycle': self.last_cleanup_cycle,
            'settings': {
                'cleanup_interval': self.memory_cleanup_interval,
                'max_cycle_history': self.max_cycle_predictions_history,
                'force_cleanup_threshold_mb': self.force_cleanup_threshold_mb
            }
        }
    
    def optimize_for_large_datasets(self):
        """대용량 데이터셋을 위한 최적화 설정"""
        self.memory_cleanup_interval = 3  # 더 자주 정리
        self.max_cycle_predictions_history = 1  # 최소 히스토리 유지
        self.force_cleanup_threshold_mb = 500.0  # 더 낮은 임계값
        
        # 예측 캐시 관리자 설정 업데이트
        for cache_manager in self.cache_managers:
            if isinstance(cache_manager, PredictionCacheManager):
                cache_manager.set_max_keep_cycles(1)
        
        logging.info("Memory manager optimized for large datasets")
    
    def get_memory_report(self) -> str:
        """메모리 사용 리포트를 생성합니다."""
        stats = self.get_memory_statistics()
        
        report = []
        report.append("=== MEMORY USAGE REPORT ===")
        
        # 메모리 사용량
        memory = stats.get('memory_usage', {})
        report.append(f"Initial Memory: {memory.get('initial_memory_mb', 0):.1f}MB")
        report.append(f"Peak Memory: {memory.get('peak_memory_mb', 0):.1f}MB")
        report.append(f"Current Memory: {memory.get('current_memory_mb', 0):.1f}MB")
        report.append(f"Memory Increase: {memory.get('memory_increase_mb', 0):.1f}MB")
        
        # 정리 작업
        report.append(f"\nCleanup Operations: {stats['cleanup_operations']}")
        report.append(f"Last Cleanup Cycle: {stats['last_cleanup_cycle']}")
        
        # 캐시 통계
        cache_stats = stats.get('cache_statistics', {})
        if cache_stats:
            report.append(f"\nCache Statistics:")
            for cache_name, cache_data in cache_stats.items():
                report.append(f"  {cache_name}: {cache_data}")
        
        report.append("=== END REPORT ===")
        
        return "\n".join(report)


# 전역 메모리 관리자 인스턴스 생성을 위한 함수
def create_memory_manager(**kwargs) -> MemoryManager:
    """메모리 관리자 인스턴스를 생성합니다."""
    return MemoryManager(**kwargs)


# 편의 함수들
def log_memory_usage(context: str = ""):
    """간편한 메모리 로깅 함수"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        rss_mb = memory_info.rss / 1024 / 1024
        percent = process.memory_percent()
        
        logging.info(f"Memory usage {context}: RSS: {rss_mb:.1f}MB, Process: {percent:.1f}%")
        
    except Exception as e:
        logging.warning(f"Failed to log memory usage: {str(e)}")