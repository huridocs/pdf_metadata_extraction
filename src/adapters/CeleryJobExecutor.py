from celery import Celery
from celery.result import AsyncResult
from trainable_entity_extractor.domain.DistributedJob import DistributedJob
from trainable_entity_extractor.domain.DistributedSubJob import DistributedSubJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.JobStatus import JobStatus
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.ports.JobExecutor import JobExecutor
from trainable_entity_extractor.ports.Logger import Logger

from config import NAME, REDIS_HOST, REDIS_PORT
from drivers.distributed_worker.distributed_gpu import train_gpu, performance_gpu, predict_gpu
from drivers.distributed_worker.distributed_no_gpu import train_no_gpu, performance_no_gpu, predict_no_gpu, upload_model
from use_cases.SampleProcessorUseCase import SampleProcessorUseCase


class CeleryJobExecutor(JobExecutor):
    def __init__(self, logger: Logger):
        super().__init__([], None, None, logger)
        self.app = Celery(NAME, broker=f"redis://{REDIS_HOST}:{REDIS_PORT}", backend=f"redis://{REDIS_HOST}:{REDIS_PORT}")

    def start_performance_evaluation(
        self, extraction_identifier: ExtractionIdentifier, distributed_sub_job: DistributedSubJob
    ):
        try:
            extractor_job = distributed_sub_job.extractor_job
            if extractor_job.gpu_needed:
                try:
                    celery_result = performance_gpu.delay(extractor_job, extractor_job.options, extractor_job.multi_value)
                except Exception:
                    celery_result = performance_no_gpu.delay(extractor_job, extractor_job.options, extractor_job.multi_value)
            else:
                celery_result = performance_no_gpu.delay(extractor_job, extractor_job.options, extractor_job.multi_value)

            distributed_sub_job.job_id = celery_result.id
            distributed_sub_job.status = JobStatus.RUNNING
        except Exception as e:
            self.logger.log(
                extraction_identifier,
                f"Performance evaluation startup failed for {distributed_sub_job.extractor_job.method_name}: {e}",
                LogSeverity.error,
            )
            distributed_sub_job.status = JobStatus.FAILURE

    def start_training(self, extraction_identifier: ExtractionIdentifier, distributed_sub_job: DistributedSubJob):
        try:
            extractor_job = distributed_sub_job.extractor_job
            if extractor_job.gpu_needed:
                try:
                    train_result = train_gpu.delay(extractor_job, extractor_job.options, extractor_job.multi_value)
                except Exception:
                    train_result = train_no_gpu.delay(extractor_job, extractor_job.options, extractor_job.multi_value)
            else:
                train_result = train_no_gpu.delay(extractor_job, extractor_job.options, extractor_job.multi_value)

            distributed_sub_job.job_id = train_result.id
            distributed_sub_job.status = JobStatus.RUNNING
        except Exception as e:
            self.logger.log(
                extraction_identifier,
                f"Training startup failed for {distributed_sub_job.extractor_job.method_name}: {e}",
                LogSeverity.error,
            )
            distributed_sub_job.status = JobStatus.FAILURE

    def start_prediction(self, extraction_identifier: ExtractionIdentifier, distributed_sub_job: DistributedSubJob):
        try:
            extractor_job = distributed_sub_job.extractor_job
            if extractor_job.gpu_needed:
                celery_result = predict_gpu.delay(extractor_job)
            else:
                celery_result = predict_no_gpu.delay(extractor_job)

            distributed_sub_job.job_id = celery_result.id
            distributed_sub_job.status = JobStatus.RUNNING
        except Exception as e:
            self.logger.log(
                extraction_identifier,
                f"Prediction startup failed for {distributed_sub_job.extractor_job.method_name}: {e}",
                LogSeverity.error,
            )
            distributed_sub_job.status = JobStatus.FAILURE

    def update_job_statuses(self, distributed_job: DistributedJob):
        for sub_job in distributed_job.sub_jobs:
            if sub_job.status not in self.get_finished_status():
                if not sub_job.job_id:
                    continue
                try:
                    celery_result = AsyncResult(sub_job.job_id, app=self.app)
                    if celery_result.state == "SUCCESS":
                        sub_job.status = JobStatus.SUCCESS
                        sub_job.result = celery_result.result
                    elif celery_result.state == "FAILURE":
                        sub_job.status = JobStatus.FAILURE
                    elif celery_result.state == "REVOKED":
                        sub_job.status = JobStatus.CANCELED
                except Exception as e:
                    self.logger.log(
                        distributed_job.extraction_identifier,
                        f"Error updating status for job {sub_job.job_id}: {e}",
                        LogSeverity.error,
                    )
                    sub_job.status = JobStatus.FAILURE

    def cancel_jobs(self, job: DistributedJob) -> None:
        for sub_job in job.sub_jobs:
            if sub_job.status not in self.get_finished_status():
                try:
                    self.app.control.revoke(sub_job.job_id, terminate=True)
                    sub_job.status = JobStatus.CANCELED
                except Exception as e:
                    self.logger.log(
                        job.extraction_identifier, f"Error canceling job {sub_job.job_id}: {e}", LogSeverity.error
                    )

    def upload_model(self, extraction_identifier: ExtractionIdentifier, extractor_job) -> bool:
        try:
            job = self.app.control.broadcast("upload_model", args=[extraction_identifier, extractor_job], reply=True)

            results = []
            for worker_reply in job:
                for worker_name, result in worker_reply.items():
                    if result and len(result) > 0:
                        success, message = result[0]
                        results.append(success)

            if not results:
                self.logger.log(
                    extraction_identifier,
                    "No workers responded to upload_model broadcast",
                    LogSeverity.error,
                )
                return False

            any_successful = any(results)
            successful_count = sum(results)
            total_workers = len(results)

            if any_successful:
                self.logger.log(
                    extraction_identifier,
                    f"Model uploaded successfully. Success rate: {successful_count}/{total_workers} workers",
                    LogSeverity.info,
                )
            else:
                self.logger.log(
                    extraction_identifier,
                    f"Model upload failed on all workers. Failed on {total_workers} workers",
                    LogSeverity.error,
                )

            return any_successful

        except Exception as e:
            self.logger.log(
                extraction_identifier,
                f"Error broadcasting upload_model: {e}",
                LogSeverity.error,
            )
            return False

    def is_extractor_cancelled(self, extractor_identifier: ExtractionIdentifier) -> bool:
        return SampleProcessorUseCase(extractor_identifier).is_extractor_cancelled()
