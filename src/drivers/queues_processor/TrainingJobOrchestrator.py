from typing import List

from trainable_entity_extractor.domain.Performance import Performance
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob

from domain.DistributedJob import DistributedJob
from domain.DistributedJobType import DistributedJobType
from domain.DistributedSubJob import DistributedSubJob
from queue_processor.QueueProcessResults import QueueProcessResults
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.use_cases.send_logs import send_logs
from drivers.queues_processor.TrainingResultBuilder import TrainingResultBuilder
from config import MODELS_DATA_PATH

# Import the new abstractions
from trainable_entity_extractor_abstractions.orchestrators.TrainingOrchestrator import TrainingOrchestrator
from adapters.CeleryJobExecutor import CeleryJobExecutor
from adapters.CeleryModelStorage import CeleryModelStorage


class TrainingJobOrchestrator:
    def __init__(
        self,
        jobs_list: List[DistributedJob],
        google_cloud_storage,
    ):
        self.jobs_list = jobs_list
        self.google_cloud_storage = google_cloud_storage

        # Initialize the abstract orchestrator with Celery implementations
        job_executor = CeleryJobExecutor()
        model_storage = CeleryModelStorage(google_cloud_storage)
        self.training_orchestrator = TrainingOrchestrator(job_executor, model_storage)

    def process_training_job(self, job: DistributedJob) -> QueueProcessResults:
        if job.type == DistributedJobType.PERFORMANCE:
            return self._process_performance_job(job)
        elif job.type == DistributedJobType.TRAIN:
            return self._process_train_job(job)
        return QueueProcessResults()

    def _process_performance_job(self, job: DistributedJob) -> QueueProcessResults:
        extraction_identifier = ExtractionIdentifier(
            run_name=job.sub_jobs[0].extractor_job.run_name,
            extraction_name=job.sub_jobs[0].extractor_job.extraction_name,
            metadata=job.task.params.metadata,
            output_path=MODELS_DATA_PATH,
        )

        # Extract extractor jobs from sub_jobs
        extractor_jobs = [sub_job.extractor_job for sub_job in job.sub_jobs]

        # Use the abstract orchestrator
        success, message, selected_job = self.training_orchestrator.process_performance_evaluation(
            extractor_jobs, job.task.params.options, job.task.params.multi_value, extraction_identifier
        )

        self.jobs_list.remove(job)

        if success:
            return TrainingResultBuilder.build_success_result(job)
        else:
            return TrainingResultBuilder.build_failure_result(job, message)

    def _process_train_job(self, job: DistributedJob) -> QueueProcessResults:
        sub_job = job.sub_jobs[0]

        extraction_identifier = ExtractionIdentifier(
            run_name=sub_job.extractor_job.run_name,
            extraction_name=sub_job.extractor_job.extraction_name,
            metadata=job.task.params.metadata,
            output_path=MODELS_DATA_PATH,
        )

        # Use the abstract orchestrator
        success, message = self.training_orchestrator.process_single_training(
            sub_job.extractor_job, job.task.params.options, job.task.params.multi_value, extraction_identifier
        )

        self.jobs_list.remove(job)

        if success:
            return TrainingResultBuilder.build_success_result(job)
        else:
            return TrainingResultBuilder.build_failure_result(job, message)
