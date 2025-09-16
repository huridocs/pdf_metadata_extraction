from typing import List
from domain.DistributedJob import DistributedJob
from queue_processor.QueueProcessResults import QueueProcessResults
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.use_cases.send_logs import send_logs
from drivers.queues_processor.PredictionJobRunner import PredictionJobRunner
from drivers.queues_processor.PredictionResultBuilder import PredictionResultBuilder


class PredictionJobOrchestrator:
    def __init__(self, jobs_list: List[DistributedJob]):
        self.jobs_list = jobs_list

    def process_prediction_job(self, job: DistributedJob) -> QueueProcessResults:
        sub_job = job.sub_jobs[0]
        runner = PredictionJobRunner(sub_job)
        runner.start_prediction_if_needed()
        status = runner.get_status()
        if status == "SUCCESS":
            return self._handle_prediction_success(job, runner)
        elif status == "FAILURE":
            return self._handle_prediction_failure(job, runner)
        return QueueProcessResults()

    def _handle_prediction_success(self, job: DistributedJob, runner: PredictionJobRunner) -> QueueProcessResults:
        success, error_message = runner.get_job_result()
        self.jobs_list.remove(job)
        if success:
            return PredictionResultBuilder.build_success_result(job)
        else:
            send_logs(ExtractionIdentifier.get_default(), f"Prediction failed: {error_message}", LogSeverity.error)
            return PredictionResultBuilder.build_failure_result(job, error_message)

    def _handle_prediction_failure(self, job: DistributedJob, runner: PredictionJobRunner) -> QueueProcessResults:
        if runner.handle_retry_if_possible():
            return QueueProcessResults()
        else:
            self.jobs_list.remove(job)
            return PredictionResultBuilder.build_max_retries_result(job)
