import os
import torch
from pydantic import ValidationError
from queue_processor.QueueProcessor import QueueProcessor
from sentry_sdk.integrations.redis import RedisIntegration
import sentry_sdk
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.send_logs import send_logs

from config import SERVICE_HOST, SERVICE_PORT, REDIS_HOST, REDIS_PORT, QUEUES_NAMES, DATA_PATH
from data.ParagraphExtractorTask import ParagraphExtractorTask
from data.TrainableEntityExtractionTask import TrainableEntityExtractionTask
from data.ResultsMessage import ResultsMessage
from Extractor import Extractor
from data.TaskType import TaskType


def restart_condition(message: dict[str, any]) -> bool:
    return TrainableEntityExtractionTask(**message).task == Extractor.CREATE_MODEL_TASK_NAME


def process(message: dict[str, any]) -> dict[str, any] | None:
    try:
        task_type = TaskType(**message)
        config_logger.info(f"New task {message}")
    except ValidationError:
        config_logger.error(f"Not a valid Redis message: {message}")
        return None

    if task_type.task in [Extractor.CREATE_MODEL_TASK_NAME, Extractor.SUGGESTIONS_TASK_NAME]:
        task = TrainableEntityExtractionTask(**message)
        result_message = get_extraction(task)
    if task_type.task == Extractor.PARAGRAPHS_EXTRACTOR_TASK_NAME:
        task = ParagraphExtractorTask(**message)
        result_message = get_extraction(task)
    else:
        task = TrainableEntityExtractionTask(**message)
        result_message = ResultsMessage(
            tenant=task.tenant,
            task=task.task,
            params=task.params,
            success=False,
            error_message="Task not found",
        )
        config_logger.error(f"Task not found: {task_to_string(task_type)}")

    return result_message.model_dump()


def get_extraction(task: TrainableEntityExtractionTask | ParagraphExtractorTask) -> ResultsMessage:
    task_calculated, error_message = Extractor.calculate_task(task)
    if task_calculated:
        data_url = None

        if task.task == Extractor.SUGGESTIONS_TASK_NAME:
            data_url = f"{SERVICE_HOST}:{SERVICE_PORT}/get_suggestions/{task.tenant}/{task.xmls.id}"

        model_results_message = ResultsMessage(
            tenant=task.tenant,
            task=task.task,
            params=task.xmls,
            success=True,
            error_message="",
            data_url=data_url,
        )
    else:
        config_logger.info(f"Error: {error_message}")
        model_results_message = ResultsMessage(
            tenant=task.tenant,
            task=task.task,
            params=task.xmls,
            success=False,
            error_message=error_message,
        )
    extraction_identifier = ExtractionIdentifier(
        run_name=task.tenant, extraction_name=task.xmls.id, metadata=task.xmls.metadata, output_path=DATA_PATH
    )
    send_logs(extraction_identifier, f"Result message: {model_results_message.to_string()}")
    return model_results_message


def task_to_string(extraction_task: TrainableEntityExtractionTask):
    extraction_dict = extraction_task.model_dump()
    if (
        "params" in extraction_dict
        and "options" in extraction_dict["params"]
        and 10 < len(extraction_dict["params"]["options"])
    ):
        extraction_dict["params"]["options"] = f'[hidden {len(extraction_dict["params"]["options"])} options]'

    return str(extraction_dict)


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
    queue_processor.start(process, restart_condition)
