import os
from pathlib import Path

from ml_cloud_connector.adapters.google_v2.GoogleCloudStorage import GoogleCloudStorage
from ml_cloud_connector.adapters.google_v2.GoogleV2Repository import GoogleV2Repository
from ml_cloud_connector.domain.ServerParameters import ServerParameters
from ml_cloud_connector.domain.ServerType import ServerType
from pydantic import ValidationError
from queue_processor.QueueProcess import QueueProcess
from queue_processor.QueueProcessResults import QueueProcessResults
from queue_processor.QueueProcessor import QueueProcessor
from sentry_sdk.integrations.redis import RedisIntegration
import sentry_sdk
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.use_cases.send_logs import send_logs

from adapters.MongoPersistenceRepository import MongoPersistenceRepository
from config import (
    SERVICE_HOST,
    SERVICE_PORT,
    REDIS_HOST,
    REDIS_PORT,
    QUEUES_NAMES,
    MODELS_DATA_PATH
)
from domain.DistributedJob import DistributedJob
from domain.DistributedJobType import DistributedJobType
from domain.DistributedSubJob import DistributedSubJob
from domain.ParagraphExtractionResultsMessage import ParagraphExtractionResultsMessage
from domain.ParagraphExtractorTask import ParagraphExtractorTask
from domain.TrainableEntityExtractionTask import TrainableEntityExtractionTask
from domain.ResultsMessage import ResultsMessage
from domain.TasksNames import TasksNames
from use_cases.ExtractorUseCase import ExtractorUseCase
from use_cases.ParagraphExtractorUseCase import ParagraphExtractorUseCase
from domain.TaskType import TaskType
from use_cases.TrainUseCase import TrainUseCase

SERVER_PARAMETERS = ServerParameters(namespace="google_v2", server_type=ServerType.METADATA_EXTRACTION)
CLOUD_PROVIDER = GoogleV2Repository(server_parameters=SERVER_PARAMETERS, service_logger=config_logger)

default_extractor_identifier = ExtractionIdentifier(extraction_name="default")




class Process(QueueProcess):

    EXTRACTOR_JOB_PATH = Path("extractor_job", "extractor_job.json")

    def __init__(self):
        super().__init__()
        try:
            server_parameters = ServerParameters(namespace="metadata_extractor",
                                                 server_type=ServerType.METADATA_EXTRACTION)
            self.google_cloud_storage = GoogleCloudStorage(server_parameters, config_logger)
            config_logger.info("Google Cloud Storage client initialized successfully")
        except Exception as e:
            config_logger.error(f"Failed to initialize Google Cloud Storage client: {e}")
            self.google_cloud_storage = None

        self.jobs: list[DistributedJob] = list()

    def process_message(self, queue_name: str, message: dict[str, any]) -> QueueProcessResults:
        try:
            task_type = TaskType(**message)
            send_logs(default_extractor_identifier, f"New task {message}")
        except ValidationError:
            send_logs(default_extractor_identifier, f"Not a valid Redis message: {message}", LogSeverity.error)
            return QueueProcessResults()

        if task_type.task == TasksNames.PARAGRAPH_EXTRACTION_TASK_NAME:
            task = ParagraphExtractorTask(**message)
            result_message = self.get_paragraphs(task)
            return QueueProcessResults(results=result_message.model_dump())

        task = TrainableEntityExtractionTask(**message)
        extraction_identifier = ExtractionIdentifier(
            run_name=task.tenant,
            extraction_name=task.params.id,
            metadata=task.params.metadata,
            output_path=MODELS_DATA_PATH,
        )
        if task_type.task == TasksNames.SUGGESTIONS_TASK_NAME:
            extractor_job = self.get_extractor_job(extraction_identifier)

            if not extractor_job:
                result_message = ResultsMessage(
                    tenant=task.tenant,
                    task=task.task,
                    params=task.params,
                    success=False,
                    error_message="Extractor job not found",
                )
                send_logs(default_extractor_identifier, f"Extractor job not found: {message}", LogSeverity.error)
                return QueueProcessResults(results=result_message.model_dump())

            distributed_job = DistributedJob(type=DistributedJobType.PREDICT,
                                             task=task,
                                             sub_jobs=[DistributedSubJob(extractor_job=extractor_job)],
                                             queue_name=queue_name)
            self.jobs.append(distributed_job)

        elif task_type.task == TasksNames.CREATE_MODEL_TASK_NAME:
            persistence_repository = MongoPersistenceRepository()
            train_use_case = TrainUseCase(extraction_identifier, persistence_repository, task.params.options, task.params.multi_value)
            distributed_job = train_use_case.get_distributed_job(task, queue_name)
            self.jobs.append(distributed_job)
        else:
            result_message = ResultsMessage(
                tenant=task.tenant,
                task=task.task,
                params=task.params,
                success=False,
                error_message="Task not found",
            )
            send_logs(default_extractor_identifier, f"Task not found: {message}", LogSeverity.error)
            return QueueProcessResults(results=result_message.model_dump())

        return self.process(queue_name)

    def process(self, queue_name: str) -> QueueProcessResults:
        for job in self.jobs:
            if job.queue_name != queue_name:
                continue

            self.process_job(job)

        return QueueProcessResults()

    @staticmethod
    def get_paragraphs(task: ParagraphExtractorTask):
        persistence_repository = MongoPersistenceRepository()
        task_calculated, error_message = ParagraphExtractorUseCase.execute_task(task, persistence_repository)

        if not task_calculated:
            send_logs(default_extractor_identifier, f"Error: {error_message}")
            return ParagraphExtractionResultsMessage(key=task.key, xmls=task.xmls, success=False, error_message=error_message)

        data_url = f"{SERVICE_HOST}:{SERVICE_PORT}/get_paragraphs_translations/{task.key}"
        return ParagraphExtractionResultsMessage(key=task.key, xmls=task.xmls, success=True, error_message="", data_url=data_url)

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







    def get_extraction(self, task: TrainableEntityExtractionTask | ParagraphExtractorTask) -> ResultsMessage:
        persistence_repository = MongoPersistenceRepository()
        task_calculated, error_message = ExtractorUseCase.execute_task(task, persistence_repository)

        model_results_message = self.get_message_for_suggestions_result(error_message, task, task_calculated)
        extraction_identifier = ExtractionIdentifier(
            run_name=task.tenant, extraction_name=task.params.id, metadata=task.params.metadata, output_path=MODELS_DATA_PATH
        )
        send_logs(extraction_identifier, f"Result message: {model_results_message.to_string()}")
        return model_results_message

    def get_message_for_suggestions_result(self, error_message, task, task_calculated):
        if task_calculated:
            if task.task == TasksNames.SUGGESTIONS_TASK_NAME:
                data_url = f"{SERVICE_HOST}:{SERVICE_PORT}/get_suggestions/{task.tenant}/{task.params.id}"
            else:
                data_url = None

            return ResultsMessage(
                tenant=task.tenant,
                task=task.task,
                params=task.params,
                success=True,
                error_message="",
                data_url=data_url,
            )

        extraction_identifier = ExtractionIdentifier(run_name=task.tenant, extraction_name="task.params.id")
        send_logs(extraction_identifier, f"Error: {error_message}")

        return ResultsMessage(
            tenant=task.tenant,
            task=task.task,
            params=task.params,
            success=False,
            error_message=error_message,
        )

    def process_job(self, job):
        pass


if __name__ == "__main__":
    try:
        sentry_sdk.init(
            os.environ.get("SENTRY_DSN"),
            traces_sample_rate=0.1,
            environment=os.environ.get("ENVIRONMENT", "development"),
            integrations=[RedisIntegration()],
        )
    except Exception:
        pass

    queues_names = QUEUES_NAMES.split(" ")
    queue_processor = QueueProcessor(REDIS_HOST, REDIS_PORT, queues_names, config_logger)

    process = Process()
    queue_processor.start(process)
