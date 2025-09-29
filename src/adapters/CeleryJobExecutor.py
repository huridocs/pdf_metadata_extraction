from typing import Tuple
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.Performance import Performance

from trainable_entity_extractor_abstractions.interfaces.JobExecutor import JobExecutor
from drivers.distributed_worker.distributed_gpu import train_gpu, performance_gpu, predict_gpu
from drivers.distributed_worker.distributed_no_gpu import train_no_gpu, performance_no_gpu, predict_no_gpu


class CeleryJobExecutor(JobExecutor):
    """Celery implementation of the JobExecutor interface"""

    def execute_training(
        self, extractor_job: TrainableEntityExtractorJob, options: list, multi_value: bool
    ) -> Tuple[bool, str]:
        """Execute a training job using Celery"""
        try:
            if extractor_job.gpu_needed:
                try:
                    train_result = train_gpu.delay(extractor_job, options, multi_value)
                except Exception:
                    train_result = train_no_gpu.delay(extractor_job, options, multi_value)
            else:
                train_result = train_no_gpu.delay(extractor_job, options, multi_value)

            # Wait for completion
            while train_result.status not in ["SUCCESS", "FAILURE"]:
                continue

            if train_result.status == "SUCCESS":
                return train_result.result
            else:
                return False, "Training task failed"

        except Exception as e:
            return False, f"Training execution failed: {e}"

    def execute_prediction(self, extractor_job: TrainableEntityExtractorJob) -> Tuple[bool, str]:
        """Execute a prediction job using Celery"""
        try:
            if extractor_job.gpu_needed:
                celery_result = predict_gpu.delay(extractor_job)
            else:
                celery_result = predict_no_gpu.delay(extractor_job)

            # Wait for completion
            while celery_result.status not in ["SUCCESS", "FAILURE"]:
                continue

            if celery_result.status == "SUCCESS":
                return celery_result.result
            else:
                return False, "Prediction task failed"

        except Exception as e:
            return False, f"Prediction execution failed: {e}"

    def execute_performance_evaluation(
        self, extractor_job: TrainableEntityExtractorJob, options: list, multi_value: bool
    ) -> Performance:
        """Execute performance evaluation using Celery"""
        try:
            if extractor_job.gpu_needed:
                try:
                    celery_result = performance_gpu.delay(extractor_job, options, multi_value)
                except Exception:
                    celery_result = performance_no_gpu.delay(extractor_job, options, multi_value)
            else:
                celery_result = performance_no_gpu.delay(extractor_job, options, multi_value)

            # Wait for completion
            while celery_result.status not in ["SUCCESS", "FAILURE"]:
                continue

            if celery_result.status == "SUCCESS":
                return celery_result.result
            else:
                return Performance()  # Return empty performance on failure

        except Exception as e:
            return Performance()  # Return empty performance on exception

    def get_job_status(self, job_id: str) -> str:
        """Get the status of a Celery job"""
        # This would need to be implemented based on how job IDs are tracked
        # For now, returning a placeholder
        return "UNKNOWN"

    def cancel_job(self, job_id: str) -> None:
        """Cancel a Celery job"""
        # This would need to be implemented based on how job IDs are tracked
        pass
