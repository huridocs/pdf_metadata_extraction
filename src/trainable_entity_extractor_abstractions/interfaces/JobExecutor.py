from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.Performance import Performance


class JobExecutor(ABC):
    """Abstract interface for executing training and prediction jobs"""

    @abstractmethod
    def execute_training(
        self, extractor_job: TrainableEntityExtractorJob, options: list, multi_value: bool
    ) -> Tuple[bool, str]:
        """Execute a training job and return success status and message"""
        pass

    @abstractmethod
    def execute_prediction(self, extractor_job: TrainableEntityExtractorJob) -> Tuple[bool, str]:
        """Execute a prediction job and return success status and message"""
        pass

    @abstractmethod
    def execute_performance_evaluation(
        self, extractor_job: TrainableEntityExtractorJob, options: list, multi_value: bool
    ) -> Performance:
        """Execute performance evaluation and return performance result"""
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> str:
        """Get the status of a running job"""
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        """Cancel a running job"""
        pass
