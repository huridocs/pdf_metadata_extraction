import os

import pymongo
import requests
import torch
from configuration import service_logger
from ml_cloud_connector.MlCloudConnector import MlCloudConnector
from ml_cloud_connector.ServerType import ServerType
from pydantic import ValidationError
from queue_processor.QueueProcessor import QueueProcessor
from sentry_sdk.integrations.redis import RedisIntegration
import sentry_sdk
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.ExtractionStatus import ExtractionStatus
from trainable_entity_extractor.send_logs import send_logs

from config import (
    SERVICE_HOST,
    SERVICE_PORT,
    REDIS_HOST,
    REDIS_PORT,
    QUEUES_NAMES,
    DATA_PATH, METADATA_EXTRACTOR_PORT, MONGO_HOST, MONGO_PORT,
)
from data.ExtractionTask import ExtractionTask
from data.ResultsMessage import ResultsMessage
from Extractor import Extractor


def restart_condition(message: dict[str, any]) -> bool:
    return ExtractionTask(**message).task == Extractor.CREATE_MODEL_TASK_NAME

def calculate_task(extraction_task: ExtractionTask) -> (bool, str):
    extractor_identifier = ExtractionIdentifier(
        run_name=extraction_task.tenant,
        extraction_name=extraction_task.params.id,
        metadata=extraction_task.params.metadata,
        output_path=DATA_PATH,
    )

    Extractor.remove_old_models(extractor_identifier)

    if extraction_task.task == Extractor.CREATE_MODEL_TASK_NAME:
        return Extractor.create_model(extractor_identifier, extraction_task.params)
    elif extraction_task.task == Extractor.SUGGESTIONS_TASK_NAME:
        return Extractor.create_suggestions(extractor_identifier, extraction_task.params)
    else:
        return False, f"Task {extraction_task.task} not recognized"


def should_wait(task):
    mongo_client = pymongo.MongoClient(f"{MONGO_HOST}:{MONGO_PORT}")
    ml_cloud_connector = MlCloudConnector(ServerType.METADATA_EXTRACTOR, service_logger)
    ip = ml_cloud_connector.get_ip()
    status = requests.get(f"http://{ip}:{METADATA_EXTRACTOR_PORT}/get_status/{task.tenant}/{task.params.id}")
    if status.status_code != 200:
        return True

    if ExtractionStatus(int(status.json())) == ExtractionStatus.PROCESSING:
        return True

    return False


def process_messages(message: dict[str, any]) -> dict[str, any] | None:
    try:
        task = ExtractionTask(**message)
        config_logger.info(f"New task {message}")
    except ValidationError:
        config_logger.error(f"Not a valid Redis message: {message}")
        return None

    if should_wait(task):
        return None

    task_calculated, error_message = calculate_task(task)

    if task_calculated:
        data_url = None

        if task.task == Extractor.SUGGESTIONS_TASK_NAME:
            data_url = f"{SERVICE_HOST}:{SERVICE_PORT}/get_suggestions/{task.tenant}/{task.params.id}"

        model_results_message = ResultsMessage(
            tenant=task.tenant,
            task=task.task,
            params=task.params,
            success=True,
            error_message="",
            data_url=data_url,
        )
    else:
        config_logger.info(f"Error: {error_message}")
        model_results_message = ResultsMessage(
            tenant=task.tenant,
            task=task.task,
            params=task.params,
            success=False,
            error_message=error_message,
        )

    extraction_identifier = ExtractionIdentifier(
        run_name=task.tenant, extraction_name=task.params.id, metadata=task.params.metadata, output_path=DATA_PATH
    )
    send_logs(extraction_identifier, f"Result message: {model_results_message.to_string()}")
    return model_results_message.model_dump()


def task_to_string(extraction_task: ExtractionTask):
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
    queue_processor.start(process_messages, restart_condition)
