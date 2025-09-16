from typing import Tuple
from domain.DistributedSubJob import DistributedSubJob
from drivers.distributed_worker.distributed_gpu import train_gpu, performance_gpu
from drivers.distributed_worker.distributed_no_gpu import train_no_gpu, performance_no_gpu


class TrainingJobRunner:
    """Handles the execution and retry logic for training jobs."""
    
    def __init__(self, sub_job: DistributedSubJob, options, multi_value: bool):
        self.sub_job = sub_job
        self.options = options
        self.multi_value = multi_value

    def start_performance_evaluation_if_needed(self) -> None:
        """Start the performance evaluation task if not already running."""
        if self.sub_job.job_id is None:
            if self.sub_job.extractor_job.gpu_needed:
                celery_result = performance_gpu.delay(
                    self.sub_job.extractor_job, self.options, self.multi_value
                )
            else:
                celery_result = performance_no_gpu.delay(
                    self.sub_job.extractor_job, self.options, self.multi_value
                )
            self.sub_job.job_id = celery_result.id

    def handle_retry_if_possible(self) -> bool:
        """Handle retry logic for failed jobs. Returns True if retry was attempted."""
        if self.sub_job.retry_count < self.sub_job.max_retries:
            self.sub_job.retry_count += 1
            if self.sub_job.extractor_job.gpu_needed:
                try:
                    celery_result = performance_gpu.delay(
                        self.sub_job.extractor_job, self.options, self.multi_value
                    )
                    self.sub_job.job_id = celery_result.id
                except Exception:
                    celery_result = performance_no_gpu.delay(
                        self.sub_job.extractor_job, self.options, self.multi_value
                    )
                    self.sub_job.job_id = celery_result.id
            else:
                celery_result = performance_no_gpu.delay(
                    self.sub_job.extractor_job, self.options, self.multi_value
                )
                self.sub_job.job_id = celery_result.id
            return True
        return False

    def execute_training(self) -> Tuple[bool, str]:
        """Execute the actual training with the selected method."""
        if self.sub_job.extractor_job.gpu_needed:
            try:
                train_result = train_gpu.delay(
                    self.sub_job.extractor_job, self.options, self.multi_value
                )
            except Exception:
                train_result = train_no_gpu.delay(
                    self.sub_job.extractor_job, self.options, self.multi_value
                )
        else:
            train_result = train_no_gpu.delay(
                self.sub_job.extractor_job, self.options, self.multi_value
            )

        while train_result.status not in ["SUCCESS", "FAILURE"]:
            continue

        if train_result.status == "SUCCESS":
            return train_result.result
        return False, "Training task failed"

    def get_performance_result(self):
        """Get the result of the performance evaluation job."""
        return self.sub_job.result()

    def get_status(self) -> str:
        """Get the current status of the performance evaluation job."""
        return self.sub_job.status()

    def cancel_job(self) -> None:
        """Cancel the current job if it's running."""
        if self.sub_job.job_id:
            self.sub_job.job.revoke(terminate=True)
