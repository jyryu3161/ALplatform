"""
Active Learning 시스템의 커스텀 예외 클래스들

각 모듈별로 특화된 예외를 정의하여 에러 처리를 체계화합니다.
"""


class ActiveLearningError(Exception):
    """Active Learning 시스템의 기본 예외 클래스"""
    pass


class DataLoadingError(ActiveLearningError):
    """데이터 로딩 관련 예외"""
    pass


class DataValidationError(ActiveLearningError):
    """데이터 검증 관련 예외"""
    pass


class MolecularProcessingError(ActiveLearningError):
    """분자 처리 관련 예외"""
    pass


class DockingError(ActiveLearningError):
    """도킹 시뮬레이션 관련 예외"""
    pass


class ModelTrainingError(ActiveLearningError):
    """모델 훈련 관련 예외"""
    pass


class PredictionError(ActiveLearningError):
    """예측 관련 예외"""
    pass


class ConfigurationError(ActiveLearningError):
    """설정 관련 예외"""
    pass


class MemoryError(ActiveLearningError):
    """메모리 관리 관련 예외"""
    pass


class SamplingError(ActiveLearningError):
    """샘플링 관련 예외"""
    pass


class ValidationError(ActiveLearningError):
    """검증 관련 예외"""
    pass


class ProcessingError(ActiveLearningError):
    """처리 관련 예외"""
    pass