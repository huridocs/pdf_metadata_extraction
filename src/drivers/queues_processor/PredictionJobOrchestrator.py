from typing import List
from domain.DistributedJob import DistributedJob
from queue_processor.QueueProcessResults import QueueProcessResults
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.use_cases.send_logs import send_logs
from drivers.queues_processor.PredictionJobRunner import PredictionJobRunner
from drivers.queues_processor.PredictionResultBuilder import PredictionResultBuilder


class PredictionJobOrchestrator:
    """Orchestrates the complete prediction job workflow."""
    
    def __init__(self, jobs_list: List[DistributedJob], default_extractor_identifier: ExtractionIdentifier):
        self.jobs_list = jobs_list
        self.default_extractor_identifier = default_extractor_identifier

    def process_prediction_job(self, job: DistributedJob) -> QueueProcessResults:
        """Main workflow for processing a prediction job."""
        sub_job = job.sub_jobs[0]
        runner = PredictionJobRunner(sub_job)
        
        # Start prediction if not already running
        runner.start_prediction_if_needed()
        
        status = runner.get_status()
        
        if status == "SUCCESS":
            return self._handle_prediction_success(job, runner)
        elif status == "FAILURE":
            return self._handle_prediction_failure(job, runner)
        
        return QueueProcessResults()

    def _handle_prediction_success(self, job: DistributedJob, runner: PredictionJobRunner) -> QueueProcessResults:
        """Handle successful prediction completion."""
        success, error_message = runner.get_job_result()
        self.jobs_list.remove(job)
        
        if success:
            return PredictionResultBuilder.build_success_result(job)
        else:
            send_logs(
                self.default_extractor_identifier, 
                f"Prediction failed: {error_message}", 
                LogSeverity.error
            )
            return PredictionResultBuilder.build_failure_result(job, error_message)

    def _handle_prediction_failure(self, job: DistributedJob, runner: PredictionJobRunner) -> QueueProcessResults:
        """Handle prediction failure with retry logic."""
        if runner.handle_retry_if_possible():
            return QueueProcessResults()  # Continue processing
        else:
            self.jobs_list.remove(job)
            return PredictionResultBuilder.build_max_retries_result(job)
