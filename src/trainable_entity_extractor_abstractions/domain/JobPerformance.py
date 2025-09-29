from pydantic import BaseModel
from trainable_entity_extractor.domain.Performance import Performance
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob

from abc import ABC, abstractmethod


class JobPerformance(BaseModel):
    """Represents the performance result of a training job"""

    extractor_job: TrainableEntityExtractorJob
    performance_result: Performance
    job_id: str

    @property
    def performance_score(self) -> float:
        """Get the performance score"""
        return self.performance_result.performance

    @property
    def is_perfect(self) -> bool:
        """Check if the performance is perfect (100%)"""
        return self.performance_score == 100.0

    @property
    def method_name(self) -> str:
        """Get the method name from the extractor job"""
        return self.extractor_job.method_name


from typing import List, Optional, Any
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob


class JobRunner(ABC):
    """Abstract base class for running distributed jobs with retry logic"""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.retry_count = 0
        self.job_id: Optional[str] = None

    @abstractmethod
    def start_job(self) -> str:
        """Start the job and return job ID"""
        pass

    @abstractmethod
    def get_status(self) -> str:
        """Get current job status"""
        pass

    @abstractmethod
    def get_result(self) -> Any:
        """Get job result when completed"""
        pass

    @abstractmethod
    def cancel(self) -> None:
        """Cancel the running job"""
        pass

    def handle_retry_if_possible(self) -> bool:
        """Handle retry logic if job failed"""
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            self.job_id = self.start_job()
            return True
        return False

    def is_completed(self) -> bool:
        """Check if job is completed (success or failure)"""
        status = self.get_status()
        return status in ["SUCCESS", "FAILURE"]

    def is_successful(self) -> bool:
        """Check if job completed successfully"""
        return self.get_status() == "SUCCESS"

    def has_failed(self) -> bool:
        """Check if job failed"""
        return self.get_status() == "FAILURE"
