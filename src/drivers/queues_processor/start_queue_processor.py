import os

import torch
from ml_cloud_connector.adapters.google_v2.GoogleV2Repository import GoogleV2Repository
from ml_cloud_connector.domain.ServerParameters import ServerParameters
from ml_cloud_connector.domain.ServerType import ServerType
from pydantic import ValidationError
from queue_processor.QueueProcessResults import QueueProcessResults
from queue_processor.QueueProcessor import QueueProcessor
from sentry_sdk.integrations.redis import RedisIntegration
import sentry_sdk
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.use_cases.send_logs import send_logs

from adapters.MongoPersistenceRepository import MongoPersistenceRepository
from config import (
    SERVICE_HOST,
    SERVICE_PORT,
    REDIS_HOST,
    REDIS_PORT,
    QUEUES_NAMES,
    DATA_PATH,
    PARAGRAPH_EXTRACTION_NAME,
    CALCULATE_MODELS_LOCALLY,
)
from domain.ParagraphExtractionResultsMessage import ParagraphExtractionResultsMessage
from domain.ParagraphExtractorTask import ParagraphExtractorTask
from domain.TrainableEntityExtractionTask import TrainableEntityExtractionTask
from domain.ResultsMessage import ResultsMessage
from use_cases.Extractor import Extractor
from domain.TaskType import TaskType


if not CALCULATE_MODELS_LOCALLY:
    SERVER_PARAMETERS = ServerParameters(namespace="google_v2", server_type=ServerType.DOCUMENT_LAYOUT_ANALYSIS)
    CLOUD_PROVIDER = GoogleV2Repository(server_parameters=SERVER_PARAMETERS, service_logger=config_logger)


def get_paragraphs(task: ParagraphExtractorTask):
    persistence_repository = MongoPersistenceRepository()
    task_calculated, error_message = Extractor.calculate_task(task, persistence_repository)

    if not task_calculated:
        config_logger.info(f"Error: {error_message}")
        return ParagraphExtractionResultsMessage(key=task.key, xmls=task.xmls, success=False, error_message=error_message)

    data_url = f"{SERVICE_HOST}:{SERVICE_PORT}/get_paragraphs_translations/{task.key}"
    return ParagraphExtractionResultsMessage(key=task.key, xmls=task.xmls, success=True, error_message="", data_url=data_url)


def process(message: dict[str, any]) -> QueueProcessResults:
    try:
        task_type = TaskType(**message)
        config_logger.info(f"New task {message}")
    except ValidationError:
        config_logger.error(f"Not a valid Redis message: {message}")
        return QueueProcessResults(results=None, delete_message=True)

    if task_type.task in [Extractor.CREATE_MODEL_TASK_NAME, Extractor.SUGGESTIONS_TASK_NAME]:
        if CALCULATE_MODELS_LOCALLY:
            task = TrainableEntityExtractionTask(**message)
            result_message = get_extraction(task)
        else:
            CLOUD_PROVIDER.start()
            return QueueProcessResults(results=None, delete_message=False, invisibility_timeout=5)

    elif task_type.task == PARAGRAPH_EXTRACTION_NAME:
        task = ParagraphExtractorTask(**message)
        result_message = get_paragraphs(task)
    else:
        task = TrainableEntityExtractionTask(**message)
        result_message = ResultsMessage(
            tenant=task.tenant,
            task=task.task,
            params=task.params,
            success=False,
            error_message="Task not found",
        )
        config_logger.error(f"Task not found: {task.model_dump()}")

    return QueueProcessResults(results=result_message.model_dump(), delete_message=True)


def get_extraction(task: TrainableEntityExtractionTask | ParagraphExtractorTask) -> ResultsMessage:
    persistence_repository = MongoPersistenceRepository()
    task_calculated, error_message = Extractor.calculate_task(task, persistence_repository)

    model_results_message = get_result_message(error_message, task, task_calculated)
    extraction_identifier = ExtractionIdentifier(
        run_name=task.tenant, extraction_name=task.params.id, metadata=task.params.metadata, output_path=DATA_PATH
    )
    send_logs(extraction_identifier, f"Result message: {model_results_message.to_string()}")
    return model_results_message


def get_result_message(error_message, task, task_calculated):
    if task_calculated:
        if task.task == Extractor.SUGGESTIONS_TASK_NAME:
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

    config_logger.info(f"Error: {error_message}")
    return ResultsMessage(
        tenant=task.tenant,
        task=task.task,
        params=task.params,
        success=False,
        error_message=error_message,
    )


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

    config_logger.info(f"Waiting for messages. Is GPU used? {torch.cuda.is_available()}")
    queues_names = QUEUES_NAMES.split(" ")
    queue_processor = QueueProcessor(REDIS_HOST, REDIS_PORT, queues_names, config_logger)
    queue_processor.start(process)
