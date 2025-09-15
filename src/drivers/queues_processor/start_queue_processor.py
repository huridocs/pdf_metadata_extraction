import os

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
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.use_cases.TrainableEntityExtractor import TrainableEntityExtractor
from trainable_entity_extractor.use_cases.send_logs import send_logs

from adapters.MongoPersistenceRepository import MongoPersistenceRepository
from config import (
    SERVICE_HOST,
    SERVICE_PORT,
    REDIS_HOST,
    REDIS_PORT,
    QUEUES_NAMES,
    MODELS_DATA_PATH,
    PARAGRAPH_EXTRACTION_NAME
)
from domain.DistributedJob import DistributedJob
from domain.DistributedJobType import DistributedJobType
from domain.DistributedSubJob import DistributedSubJob
from domain.ParagraphExtractionResultsMessage import ParagraphExtractionResultsMessage
from domain.ParagraphExtractorTask import ParagraphExtractorTask
from domain.TrainableEntityExtractionTask import TrainableEntityExtractionTask
from domain.ResultsMessage import ResultsMessage
from use_cases.ExtractorUseCase import ExtractorUseCase
from use_cases.SampleProcessorUseCase import SampleProcessorUseCase
from domain.TaskType import TaskType


SERVER_PARAMETERS = ServerParameters(namespace="google_v2", server_type=ServerType.METADATA_EXTRACTION)
CLOUD_PROVIDER = GoogleV2Repository(server_parameters=SERVER_PARAMETERS, service_logger=config_logger)

default_extractor_identifier = ExtractionIdentifier(extraction_name="default")


class Process(QueueProcess):
    def __init__(self):
        super().__init__()
        self.jobs: list[DistributedJob] = list()

    def process_message(self, queue_name: str, message: dict[str, any]) -> QueueProcessResults:
        try:
            task_type = TaskType(**message)
            send_logs(default_extractor_identifier, f"New task {message}")
        except ValidationError:
            send_logs(default_extractor_identifier, f"Not a valid Redis message: {message}", LogSeverity.error)
            return QueueProcessResults()

        if task_type.task == PARAGRAPH_EXTRACTION_NAME:
            task = ParagraphExtractorTask(**message)
            result_message = self.get_paragraphs(task)
            return QueueProcessResults(results=result_message.model_dump())

        task = TrainableEntityExtractionTask(**message)

        if task_type.task not in [ExtractorUseCase.CREATE_MODEL_TASK_NAME, ExtractorUseCase.SUGGESTIONS_TASK_NAME]:
            result_message = ResultsMessage(
                tenant=task.tenant,
                task=task.task,
                params=task.params,
                success=False,
                error_message="Task not found",
            )
            send_logs(default_extractor_identifier, f"Task not found: {message}", LogSeverity.error)
            return QueueProcessResults(results=result_message.model_dump())

        CLOUD_PROVIDER.start()

        # Create extraction identifier and data for getting distributed jobs
        extraction_identifier = ExtractionIdentifier(
            run_name=task.tenant,
            extraction_name=task.params.id,
            metadata=task.params.metadata,
            output_path=MODELS_DATA_PATH,
        )

        persistence_repository = MongoPersistenceRepository()
        sample_processor = SampleProcessorUseCase(extraction_identifier, persistence_repository)
        samples = sample_processor.get_training_samples()

        options = task.params.options if task.params.options else extraction_identifier.get_options()
        multi_value = task.params.multi_value

        extraction_data = ExtractionData(
            samples=samples,
            options=options,
            multi_value=multi_value,
            extraction_identifier=extraction_identifier,
        )

        # Get distributed jobs from trainable entity extractor
        trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier)
        extractor_jobs = trainable_entity_extractor.get_distributed_jobs(extraction_data)

        # Create distributed sub jobs
        sub_jobs = [
            DistributedSubJob(
                job_id=None,  # Will be set when job is submitted to Celery
                extractor_job=extractor_job
            )
            for extractor_job in extractor_jobs
        ]

        if task_type.task == ExtractorUseCase.SUGGESTIONS_TASK_NAME:
            job = DistributedJob(
                type=DistributedJobType.PREDICT,
                task_message=task,
                sub_jobs=sub_jobs,
                queue_name=queue_name
            )
            self.jobs.append(job)
        else:  # CREATE_MODEL_TASK_NAME
            job = DistributedJob(
                type=DistributedJobType.TRAIN,
                task_message=task,
                sub_jobs=sub_jobs,
                queue_name=queue_name
            )
            self.jobs.append(job)

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
        task_calculated, error_message = ExtractorUseCase.execute_task(task, persistence_repository)

        if not task_calculated:
            send_logs(default_extractor_identifier, f"Error: {error_message}")
            return ParagraphExtractionResultsMessage(key=task.key, xmls=task.xmls, success=False, error_message=error_message)

        data_url = f"{SERVICE_HOST}:{SERVICE_PORT}/get_paragraphs_translations/{task.key}"
        return ParagraphExtractionResultsMessage(key=task.key, xmls=task.xmls, success=True, error_message="", data_url=data_url)

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
            if task.task == ExtractorUseCase.SUGGESTIONS_TASK_NAME:
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
