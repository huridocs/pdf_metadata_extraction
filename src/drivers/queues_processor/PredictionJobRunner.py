from typing import Tuple
from domain.DistributedSubJob import DistributedSubJob
from drivers.distributed_worker.distributed_gpu import predict_gpu
from drivers.distributed_worker.distributed_no_gpu import predict_no_gpu


class PredictionJobRunner:
    def __init__(self, sub_job: DistributedSubJob):
        self.sub_job = sub_job

    def start_prediction_if_needed(self) -> None:
        if self.sub_job.job_id is None:
            if self.sub_job.extractor_job.gpu_needed:
                celery_result = predict_gpu.delay(self.sub_job.extractor_job)
            else:
                celery_result = predict_no_gpu.delay(self.sub_job.extractor_job)
            self.sub_job.job_id = celery_result.id

    def handle_retry_if_possible(self) -> bool:
        if self.sub_job.retry_count < self.sub_job.max_retries:
            self.sub_job.retry_count += 1
            if self.sub_job.extractor_job.gpu_needed:
                celery_result = predict_gpu.delay(self.sub_job.extractor_job)
            else:
                celery_result = predict_no_gpu.delay(self.sub_job.extractor_job)
            self.sub_job.job_id = celery_result.id
            return True
        return False

    def get_job_result(self) -> Tuple[bool, str]:
        return self.sub_job.result()

    def get_status(self) -> str:
        return self.sub_job.status()
