from pathlib import Path
from typing import Any

from ml_cloud_connector.adapters.google_v2.GoogleCloudStorage import GoogleCloudStorage
from ml_cloud_connector.adapters.google_v2.GoogleV2Repository import GoogleV2Repository
from ml_cloud_connector.domain.ServerParameters import ServerParameters
from ml_cloud_connector.domain.ServerType import ServerType
from pydantic import ValidationError
from queue_processor.QueueProcess import QueueProcess
from queue_processor.QueueProcessResults import QueueProcessResults
from trainable_entity_extractor.config import config_logger, EXTRACTOR_JOB_PATH
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.DistributedJob import DistributedJob
from trainable_entity_extractor.domain.JobType import JobType
from trainable_entity_extractor.domain.DistributedSubJob import DistributedSubJob
from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger
from trainable_entity_extractor.use_cases.OrchestratorUseCase import OrchestratorUseCase

from adapters.MongoPersistenceRepository import MongoPersistenceRepository
from adapters.CeleryJobExecutor import CeleryJobExecutor
from adapters.CloudModelStorage import CloudModelStorage
from config import SERVICE_HOST, SERVICE_PORT, MODELS_DATA_PATH
from domain.ParagraphExtractionResultsMessage import ParagraphExtractionResultsMessage
from domain.ParagraphExtractorTask import ParagraphExtractorTask
from domain.ResultsMessage import ResultsMessage
from domain.TasksNames import TasksNames
from domain.TaskType import TaskType
from domain.TrainableEntityExtractionTask import TrainableEntityExtractionTask
from use_cases.ParagraphExtractorUseCase import ParagraphExtractorUseCase
from use_cases.TrainUseCase import TrainUseCase
from drivers.queues_processor.PredictionResultBuilder import PredictionResultBuilder
from drivers.queues_processor.TrainingResultBuilder import TrainingResultBuilder


class MetadataExtractorQueueProcessor(QueueProcess):
    def __init__(self):
        super().__init__()
        self.logger = ExtractorLogger()

        try:
            server_parameters = ServerParameters(namespace="metadata_extractor", server_type=ServerType.METADATA_EXTRACTION)
            self.google_cloud_storage = GoogleCloudStorage(server_parameters, config_logger)
            self.logger.log(ExtractionIdentifier.get_default(), "Google Cloud Storage client initialized successfully")
        except Exception as e:
            self.logger.log(
                ExtractionIdentifier.get_default(),
                f"Failed to initialize Google Cloud Storage client: {e}",
                LogSeverity.error,
                e,
            )
            self.google_cloud_storage = None

        self.model_storage = CloudModelStorage(self.google_cloud_storage, self.logger)
        self.job_executor = CeleryJobExecutor()
        self.orchestrator = OrchestratorUseCase(self.job_executor, self.logger)
        server_parameters = ServerParameters(namespace="google_v2", server_type=ServerType.METADATA_EXTRACTION)
        self.cloud_provider = GoogleV2Repository(server_parameters=server_parameters, service_logger=config_logger)

    def process_message(self, queue_name: str, message: dict[str, Any]) -> QueueProcessResults:
        task_type = self._validate_and_parse_message(message)
        if not task_type:
            return QueueProcessResults()

        if task_type.task == TasksNames.PARAGRAPH_EXTRACTION_TASK_NAME:
            return self._handle_paragraph_extraction_task(message)

        return self._handle_trainable_entity_extraction_task(queue_name, message)

    def process(self, queue_name: str) -> QueueProcessResults:
        for distributed_job in self.orchestrator.distributed_jobs:
            if distributed_job.domain_name != queue_name:
                continue

            return self._process_job_with_orchestrator(distributed_job)

        return QueueProcessResults()

    def _handle_trainable_entity_extraction_task(self, queue_name: str, message: dict[str, Any]) -> QueueProcessResults:
        task = TrainableEntityExtractionTask(**message)
        extraction_identifier = self._create_extraction_identifier(task)

        if task.task == TasksNames.SUGGESTIONS_TASK_NAME:
            self._handle_suggestions_task(task, extraction_identifier, queue_name)
        elif task.task == TasksNames.CREATE_MODEL_TASK_NAME:
            self._handle_create_model_task(task, extraction_identifier, queue_name)
        else:
            return self._create_task_not_found_result(task)

        return self.process(queue_name)

    def _process_job_with_orchestrator(self, job: DistributedJob) -> QueueProcessResults:
        """Process a distributed job using OrchestratorUseCase"""
        try:
            if job.sub_jobs and job.sub_jobs[0].extractor_job.gpu_needed:
                self.cloud_provider.start()

            success, message = self.orchestrator.process_job(job)

            if job.type == JobType.PREDICT:
                if success:
                    return PredictionResultBuilder.build_success_result(job)
                else:
                    return PredictionResultBuilder.build_failure_result(job, message)
            elif job.type in [JobType.TRAIN, JobType.PERFORMANCE]:
                if success:
                    return TrainingResultBuilder.build_success_result(job)
                else:
                    if "in progress" in message.lower():
                        return QueueProcessResults()  # Continue processing
                    else:
                        return TrainingResultBuilder.build_failure_result(job, message)

            return QueueProcessResults()

        except Exception as e:
            self.logger.log(job.extraction_identifier, f"Error processing job with orchestrator: {e}", LogSeverity.error)
            if job.type == JobType.PREDICT:
                return PredictionResultBuilder.build_failure_result(job, str(e))
            else:
                return TrainingResultBuilder.build_failure_result(job, str(e))

    def get_extractor_job(self, extraction_identifier: ExtractionIdentifier) -> TrainableEntityExtractorJob | None:
        path = Path(extraction_identifier.get_path(), EXTRACTOR_JOB_PATH)

        if not path.exists():
            if not self.model_storage.download_model(extraction_identifier):
                self.logger.log(extraction_identifier, "Failed to download model from cloud storage", LogSeverity.error)
                return None

        try:
            with open(path, "r", encoding="utf-8") as file:
                job_data = file.read()
                extractor_job = TrainableEntityExtractorJob.model_validate_json(job_data)
                return extractor_job
        except Exception as e:
            self.logger.log(extraction_identifier, f"Error reading extractor job file: {e}", LogSeverity.error)
            return None

    def _validate_and_parse_message(self, message: dict[str, Any]) -> TaskType | None:
        try:
            task_type = TaskType(**message)
            self.logger.log(ExtractionIdentifier.get_default(), f"New task {message}")
            return task_type
        except ValidationError:
            self.logger.log(ExtractionIdentifier.get_default(), f"Not a valid Redis message: {message}", LogSeverity.error)
            return None

    def _handle_paragraph_extraction_task(self, message: dict[str, Any]) -> QueueProcessResults:
        task = ParagraphExtractorTask(**message)
        persistence_repository = MongoPersistenceRepository()
        task_calculated, error_message = ParagraphExtractorUseCase.execute_task(task, persistence_repository)

        if not task_calculated:
            self.logger.log(ExtractionIdentifier.get_default(), f"Error: {error_message}", LogSeverity.error)
            result_message = ParagraphExtractionResultsMessage(
                key=task.key, xmls=task.xmls, success=False, error_message=error_message
            )
            return QueueProcessResults(results=result_message.model_dump())

        data_url = f"{SERVICE_HOST}:{SERVICE_PORT}/get_paragraphs_translations/{task.key}"

        result_message = ParagraphExtractionResultsMessage(
            key=task.key, xmls=task.xmls, success=True, error_message="", data_url=data_url
        )
        return QueueProcessResults(results=result_message.model_dump())

    @staticmethod
    def _create_extraction_identifier(task: TrainableEntityExtractionTask) -> ExtractionIdentifier:
        return ExtractionIdentifier(
            run_name=task.tenant,
            extraction_name=task.params.id,
            metadata=task.params.metadata,
            output_path=MODELS_DATA_PATH,
        )

    def _handle_suggestions_task(
        self, task: TrainableEntityExtractionTask, extraction_identifier: ExtractionIdentifier, queue_name: str
    ):
        extractor_job = self.get_extractor_job(extraction_identifier)

        if not extractor_job:
            return self._create_extractor_not_found_result(task)

        distributed_job = DistributedJob(
            type=JobType.PREDICT,
            sub_jobs=[DistributedSubJob(extractor_job=extractor_job)],
            domain_name=queue_name,
            extraction_identifier=extraction_identifier,
        )
        # Add job to orchestrator instead of self.jobs
        self.orchestrator.distributed_jobs.append(distributed_job)

    def _handle_create_model_task(
        self, task: TrainableEntityExtractionTask, extraction_identifier: ExtractionIdentifier, queue_name: str
    ):
        train_use_case = TrainUseCase(extraction_identifier, task.params.options, task.params.multi_value)
        distributed_job = train_use_case.get_distributed_job(task, queue_name)
        # Add job to orchestrator instead of self.jobs
        self.orchestrator.distributed_jobs.append(distributed_job)

    def _create_extractor_not_found_result(self, task: TrainableEntityExtractionTask) -> QueueProcessResults:
        result_message = ResultsMessage(
            tenant=task.tenant,
            task=task.task,
            params=task.params,
            success=False,
            error_message="Extractor job not found",
        )
        self.logger.log(
            ExtractionIdentifier.get_default(), f"Extractor job not found: {task.model_dump()}", LogSeverity.error
        )
        return QueueProcessResults(results=result_message.model_dump())

    def _create_task_not_found_result(self, task: TrainableEntityExtractionTask) -> QueueProcessResults:
        result_message = ResultsMessage(
            tenant=task.tenant,
            task=task.task,
            params=task.params,
            success=False,
            error_message="Task not found",
        )
        self.logger.log(ExtractionIdentifier.get_default(), f"Task not found: {task.model_dump()}", LogSeverity.error)
        return QueueProcessResults(results=result_message.model_dump())
