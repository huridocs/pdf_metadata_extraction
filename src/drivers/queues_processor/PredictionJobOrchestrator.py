from typing import List
from domain.DistributedJob import DistributedJob
from queue_processor.QueueProcessResults import QueueProcessResults
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.use_cases.send_logs import send_logs
from drivers.queues_processor.PredictionResultBuilder import PredictionResultBuilder
from config import MODELS_DATA_PATH

# Import the new abstractions
from trainable_entity_extractor_abstractions.orchestrators.PredictionOrchestrator import PredictionOrchestrator
from adapters.CeleryJobExecutor import CeleryJobExecutor
from adapters.CeleryModelStorage import CeleryModelStorage


class PredictionJobOrchestrator:
    def __init__(self, jobs_list: List[DistributedJob], google_cloud_storage=None):
        self.jobs_list = jobs_list

        # Initialize the abstract orchestrator with Celery implementations
        job_executor = CeleryJobExecutor()
        model_storage = CeleryModelStorage(google_cloud_storage)
        self.prediction_orchestrator = PredictionOrchestrator(job_executor, model_storage)

    def process_prediction_job(self, job: DistributedJob) -> QueueProcessResults:
        sub_job = job.sub_jobs[0]

        extraction_identifier = ExtractionIdentifier(
            run_name=sub_job.extractor_job.run_name,
            extraction_name=sub_job.extractor_job.extraction_name,
            metadata=job.task.params.metadata,
            output_path=MODELS_DATA_PATH,
        )

        # Reset retry count for new job
        self.prediction_orchestrator.reset_retry_count()

        # Use the abstract orchestrator with model completion checking
        success, message, should_retry = self.prediction_orchestrator.process_prediction(
            sub_job.extractor_job, extraction_identifier, wait_for_model=True
        )

        if should_retry:
            # Don't remove job from list, let it retry in next iteration
            send_logs(extraction_identifier, "Prediction will be retried in next iteration")
            return QueueProcessResults()

        # Remove job from list as it's completed (success or failure)
        self.jobs_list.remove(job)

        if success:
            return PredictionResultBuilder.build_success_result(job)
        else:
            return PredictionResultBuilder.build_failure_result(job, message)
