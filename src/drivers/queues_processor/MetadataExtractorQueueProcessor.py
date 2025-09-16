from pathlib import Path
from typing import Any

from ml_cloud_connector.adapters.google_v2.GoogleCloudStorage import GoogleCloudStorage
from ml_cloud_connector.adapters.google_v2.GoogleV2Repository import GoogleV2Repository
from ml_cloud_connector.domain.ServerParameters import ServerParameters
from ml_cloud_connector.domain.ServerType import ServerType
from pydantic import ValidationError
from queue_processor.QueueProcess import QueueProcess
from queue_processor.QueueProcessResults import QueueProcessResults
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.use_cases.send_logs import send_logs
from drivers.distributed_worker.distributed_gpu import predict_gpu, train_gpu, performance_gpu
from drivers.distributed_worker.distributed_no_gpu import predict_no_gpu, train_no_gpu, performance_no_gpu, upload_model

from adapters.MongoPersistenceRepository import MongoPersistenceRepository
from config import SERVICE_HOST, SERVICE_PORT, MODELS_DATA_PATH
from domain.DistributedJob import DistributedJob
from domain.DistributedJobType import DistributedJobType
from domain.DistributedSubJob import DistributedSubJob
from domain.ParagraphExtractionResultsMessage import ParagraphExtractionResultsMessage
from domain.ParagraphExtractorTask import ParagraphExtractorTask
from domain.TrainableEntityExtractionTask import TrainableEntityExtractionTask
from domain.ResultsMessage import ResultsMessage
from domain.TasksNames import TasksNames
from use_cases.ParagraphExtractorUseCase import ParagraphExtractorUseCase
from domain.TaskType import TaskType
from use_cases.TrainUseCase import TrainUseCase
from drivers.queues_processor.PredictionJobOrchestrator import PredictionJobOrchestrator
from drivers.queues_processor.TrainingJobOrchestrator import TrainingJobOrchestrator


class MetadataExtractorQueueProcessor(QueueProcess):

    EXTRACTOR_JOB_PATH = Path("extractor_job", "extractor_job.json")
    DEFAULT_EXTRACTOR_IDENTIFIER = ExtractionIdentifier(extraction_name="default")
    SERVER_PARAMETERS = ServerParameters(namespace="google_v2", server_type=ServerType.METADATA_EXTRACTION)
    CLOUD_PROVIDER = GoogleV2Repository(server_parameters=SERVER_PARAMETERS, service_logger=config_logger)

    def __init__(self):
        super().__init__()
        try:
            server_parameters = ServerParameters(namespace="metadata_extractor", server_type=ServerType.METADATA_EXTRACTION)
            self.google_cloud_storage = GoogleCloudStorage(server_parameters, config_logger)
            config_logger.info("Google Cloud Storage client initialized successfully")
        except Exception as e:
            config_logger.error(f"Failed to initialize Google Cloud Storage client: {e}")
            self.google_cloud_storage = None
        self.jobs: list[DistributedJob] = list()

    def process_message(self, queue_name: str, message: dict[str, Any]) -> QueueProcessResults:
        task_type = self._validate_and_parse_message(message)
        if not task_type:
            return QueueProcessResults()

        if task_type.task == TasksNames.PARAGRAPH_EXTRACTION_TASK_NAME:
            return self._handle_paragraph_extraction_task(message)

        return self._handle_trainable_entity_task(message, task_type, queue_name)

    def process(self, queue_name: str) -> QueueProcessResults:
        for job in self.jobs:
            if job.queue_name != queue_name:
                continue

            return self.process_job(job)

        return QueueProcessResults()

    def process_job(self, job: DistributedJob) -> QueueProcessResults:
        self.CLOUD_PROVIDER.start()
        if job.type == DistributedJobType.PREDICT:
            orchestrator = PredictionJobOrchestrator(self.jobs, self.DEFAULT_EXTRACTOR_IDENTIFIER)
            return orchestrator.process_prediction_job(job)
        elif job.type == DistributedJobType.TRAIN:
            orchestrator = TrainingJobOrchestrator(
                self.jobs,
                self.DEFAULT_EXTRACTOR_IDENTIFIER,
                self.google_cloud_storage,
                self.EXTRACTOR_JOB_PATH
            )
            return orchestrator.process_training_job(job)

        return QueueProcessResults()

    def get_extractor_job(self, extraction_identifier: ExtractionIdentifier) -> TrainableEntityExtractorJob | None:
        path = Path(extraction_identifier.get_path(), self.EXTRACTOR_JOB_PATH)

        if not path.exists():
            self.google_cloud_storage.copy_from_cloud(
                path.parent, Path(MODELS_DATA_PATH, extraction_identifier.run_name, self.EXTRACTOR_JOB_PATH.parent)
            )

        try:
            with open(path, "r", encoding="utf-8") as file:
                job_data = file.read()
                extractor_job = TrainableEntityExtractorJob.model_validate_json(job_data)
                return extractor_job
        except Exception as e:
            send_logs(extraction_identifier, f"Error reading extractor job file: {e}", LogSeverity.error)
            return None

    def _validate_and_parse_message(self, message: dict[str, Any]) -> TaskType | None:
        try:
            task_type = TaskType(**message)
            send_logs(self.DEFAULT_EXTRACTOR_IDENTIFIER, f"New task {message}")
            return task_type
        except ValidationError:
            send_logs(self.DEFAULT_EXTRACTOR_IDENTIFIER, f"Not a valid Redis message: {message}", LogSeverity.error)
            return None

    def _handle_paragraph_extraction_task(self, message: dict[str, Any]) -> QueueProcessResults:
        task = ParagraphExtractorTask(**message)
        persistence_repository = MongoPersistenceRepository()
        task_calculated, error_message = ParagraphExtractorUseCase.execute_task(task, persistence_repository)

        if not task_calculated:
            send_logs(self.DEFAULT_EXTRACTOR_IDENTIFIER, f"Error: {error_message}")
            result_message = ParagraphExtractionResultsMessage(
                key=task.key, xmls=task.xmls, success=False, error_message=error_message
            )
            return QueueProcessResults(results=result_message.model_dump())

        data_url = f"{SERVICE_HOST}:{SERVICE_PORT}/get_paragraphs_translations/{task.key}"

        result_message = ParagraphExtractionResultsMessage(
            key=task.key, xmls=task.xmls, success=True, error_message="", data_url=data_url
        )
        return QueueProcessResults(results=result_message.model_dump())

    def _handle_trainable_entity_task(
        self, message: dict[str, Any], task_type: TaskType, queue_name: str
    ) -> QueueProcessResults:
        task = TrainableEntityExtractionTask(**message)
        extraction_identifier = self._create_extraction_identifier(task)

        if task_type.task == TasksNames.SUGGESTIONS_TASK_NAME:
            return self._handle_suggestions_task(task, extraction_identifier, queue_name)
        elif task_type.task == TasksNames.CREATE_MODEL_TASK_NAME:
            return self._handle_create_model_task(task, extraction_identifier, queue_name)
        else:
            return self._create_task_not_found_result(task)

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
    ) -> QueueProcessResults:
        extractor_job = self.get_extractor_job(extraction_identifier)

        if not extractor_job:
            return self._create_extractor_not_found_result(task)

        distributed_job = DistributedJob(
            type=DistributedJobType.PREDICT,
            task=task,
            sub_jobs=[DistributedSubJob(extractor_job=extractor_job)],
            queue_name=queue_name,
        )
        self.jobs.append(distributed_job)
        return self.process(queue_name)

    def _handle_create_model_task(
        self, task: TrainableEntityExtractionTask, extraction_identifier: ExtractionIdentifier, queue_name: str
    ) -> QueueProcessResults:
        persistence_repository = MongoPersistenceRepository()
        train_use_case = TrainUseCase(
            extraction_identifier, persistence_repository, task.params.options, task.params.multi_value
        )
        distributed_job = train_use_case.get_distributed_job(task, queue_name)
        self.jobs.append(distributed_job)
        return self.process(queue_name)

    def _create_extractor_not_found_result(self, task: TrainableEntityExtractionTask) -> QueueProcessResults:
        result_message = ResultsMessage(
            tenant=task.tenant,
            task=task.task,
            params=task.params,
            success=False,
            error_message="Extractor job not found",
        )
        send_logs(self.DEFAULT_EXTRACTOR_IDENTIFIER, f"Extractor job not found: {task.model_dump()}", LogSeverity.error)
        return QueueProcessResults(results=result_message.model_dump())

    def _create_task_not_found_result(self, task: TrainableEntityExtractionTask) -> QueueProcessResults:
        result_message = ResultsMessage(
            tenant=task.tenant,
            task=task.task,
            params=task.params,
            success=False,
            error_message="Task not found",
        )
        send_logs(self.DEFAULT_EXTRACTOR_IDENTIFIER, f"Task not found: {task.model_dump()}", LogSeverity.error)
        return QueueProcessResults(results=result_message.model_dump())
