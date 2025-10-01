from typing import Any

from ml_cloud_connector.adapters.google_v2.GoogleCloudStorage import GoogleCloudStorage
from ml_cloud_connector.adapters.google_v2.GoogleV2Repository import GoogleV2Repository
from ml_cloud_connector.domain.ServerParameters import ServerParameters
from ml_cloud_connector.domain.ServerType import ServerType
from pydantic import ValidationError
from queue_processor.QueueProcess import QueueProcess
from queue_processor.QueueProcessResults import QueueProcessResults
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import (
    PdfToMultiOptionExtractor,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_text_extractor.PdfToTextExtractor import PdfToTextExtractor
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionExtractor import (
    TextToMultiOptionExtractor,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.TextToTextExtractor import TextToTextExtractor
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.JobProcessingResult import JobProcessingResult
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.DistributedJob import DistributedJob
from trainable_entity_extractor.domain.JobType import JobType
from trainable_entity_extractor.domain.DistributedSubJob import DistributedSubJob
from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger
from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase
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
from use_cases.GetPerformanceJobUseCase import GetPerformanceJobUseCase
from use_cases.ParagraphExtractorUseCase import ParagraphExtractorUseCase
from drivers.queues_processor.PredictionResultBuilder import PredictionResultBuilder
from drivers.queues_processor.TrainingResultBuilder import TrainingResultBuilder


class MetadataExtractorQueueProcessor(QueueProcess):
    def __init__(self):
        super().__init__()
        self.logger = ExtractorLogger()
        self.google_cloud_storage = None

        if GoogleCloudStorage.could_be_configured():
            server_parameters = ServerParameters(namespace="metadata_extractor", server_type=ServerType.METADATA_EXTRACTION)
            self.google_cloud_storage = GoogleCloudStorage(server_parameters, config_logger)
            self.logger.log(ExtractionIdentifier.get_default(), "Google Cloud Storage client initialized successfully")

        self.model_storage = CloudModelStorage(self.google_cloud_storage, self.logger)
        self.job_executor = CeleryJobExecutor(self.logger)
        self.orchestrator = OrchestratorUseCase(self.job_executor, self.logger)
        server_parameters = ServerParameters(namespace="google_v2", server_type=ServerType.METADATA_EXTRACTION)
        try:
            self.cloud_provider = GoogleV2Repository(server_parameters=server_parameters, service_logger=config_logger)
        except:
            self.cloud_provider = None
        self.extractors: list[type[ExtractorBase]] = [
            PdfToMultiOptionExtractor,
            TextToMultiOptionExtractor,
            PdfToTextExtractor,
            TextToTextExtractor,
        ]

    def process_message(self, queue_name: str, message: dict[str, Any]) -> QueueProcessResults:
        task_type = self._validate_and_parse_message(message)
        if not task_type:
            return QueueProcessResults()

        if task_type.task == TasksNames.PARAGRAPH_EXTRACTION_TASK_NAME:
            return self._handle_paragraph_extraction_task(message)

        return self._handle_trainable_entity_extraction_task(queue_name, message)

    def process(self, queue_name: str) -> QueueProcessResults:
        job_processing_result, distributed_job = self.orchestrator.execute_job_for_domain(queue_name)

        if job_processing_result.gpu_needed and self.cloud_provider:
            self.cloud_provider.start()

        if not job_processing_result.finished:
            return QueueProcessResults()

        return self._convert_orchestrator_result_to_queue_result(job_processing_result, distributed_job)

    def _convert_orchestrator_result_to_queue_result(
        self, job_processing_result: JobProcessingResult, processed_job: DistributedJob
    ) -> QueueProcessResults:
        if not processed_job:
            return QueueProcessResults()

        try:
            if processed_job.type == JobType.PREDICT:
                if job_processing_result.success:
                    return PredictionResultBuilder.build_success_result(processed_job)
                else:
                    return PredictionResultBuilder.build_failure_result(processed_job, job_processing_result.error_message)
            elif processed_job.type in [JobType.TRAIN, JobType.PERFORMANCE]:
                if job_processing_result.success:
                    return TrainingResultBuilder.build_success_result(processed_job)
                else:
                    return TrainingResultBuilder.build_failure_result(processed_job, job_processing_result.error_message)
        except Exception as e:
            self.logger.log(
                processed_job.extraction_identifier, f"Error converting orchestrator result: {e}", LogSeverity.error
            )
            if processed_job.type == JobType.PREDICT:
                return PredictionResultBuilder.build_failure_result(processed_job, str(e))
            else:
                return TrainingResultBuilder.build_failure_result(processed_job, str(e))

        return QueueProcessResults()

    def _handle_trainable_entity_extraction_task(self, queue_name: str, message: dict[str, Any]) -> QueueProcessResults:
        task = TrainableEntityExtractionTask(**message)
        extraction_identifier = self._create_extraction_identifier(task)

        if task.task == TasksNames.SUGGESTIONS_TASK_NAME:
            return self._handle_suggestions_task(task, extraction_identifier, queue_name)
        elif task.task == TasksNames.CREATE_MODEL_TASK_NAME:
            return self._handle_create_model_task(task, extraction_identifier, queue_name)
        else:
            return self._create_task_not_found_result(task)

    def _handle_suggestions_task(
        self, task: TrainableEntityExtractionTask, extraction_identifier: ExtractionIdentifier, queue_name: str
    ) -> QueueProcessResults:
        extractor_job = self.model_storage.get_extractor_job(extraction_identifier)

        if not extractor_job:
            return self._create_extractor_not_found_result(task)

        distributed_job = DistributedJob(
            type=JobType.PREDICT,
            sub_jobs=[DistributedSubJob(extractor_job=extractor_job)],
            domain_name=queue_name,
            extraction_identifier=extraction_identifier,
        )
        self.orchestrator.add_job(distributed_job)
        return self.process(queue_name)

    def _handle_create_model_task(
        self, task: TrainableEntityExtractionTask, extraction_identifier: ExtractionIdentifier, queue_name: str
    ):
        get_performance_job_use_case = GetPerformanceJobUseCase(
            extraction_identifier, task.params.options, task.params.multi_value
        )
        distributed_job = get_performance_job_use_case.get_distributed_job(queue_name, self.extractors, self.logger)
        self.orchestrator.add_job(distributed_job)
        return self.process(queue_name)

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
