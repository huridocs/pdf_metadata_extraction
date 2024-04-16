import os
from time import sleep

import redis
import torch
from pydantic import ValidationError
from rsmq.consumer import RedisSMQConsumer
from rsmq import RedisSMQ, cmd
from sentry_sdk.integrations.redis import RedisIntegration
import sentry_sdk

from config import (
    config_logger,
    SERVICE_HOST,
    SERVICE_PORT,
    REDIS_HOST,
    REDIS_PORT,
    TASK_QUEUE_NAME,
    RESULTS_QUEUE_NAME,
    logs_queue,
)
from data.ExtractionTask import ExtractionTask
from data.ResultsMessage import ResultsMessage
from Extractor import Extractor


class QueueProcessor:
    def __init__(self):
        config_logger.info("The PDF Metadata Extractor has been started")

        self.task_queue = RedisSMQ(
            host=REDIS_HOST,
            port=REDIS_PORT,
            qname=TASK_QUEUE_NAME,
        )

        self.results_queue = RedisSMQ(
            host=REDIS_HOST,
            port=REDIS_PORT,
            qname=RESULTS_QUEUE_NAME,
        )

    def process(self, id, message, rc, ts):
        try:
            task = ExtractionTask(**message)
            config_logger.info(f"New task {task.model_dump()}")
        except ValidationError:
            config_logger.error(f"Not a valid Redis message: {message}")
            return True

        self.log_process_information(message)

        task_calculated, error_message = Extractor.calculate_task(task)
        if error_message:
            config_logger.info(f"Error: {error_message}")

        if not task_calculated:
            model_results_message = ResultsMessage(
                tenant=task.tenant,
                task=task.task,
                params=task.params,
                success=False,
                error_message=error_message,
            )

        else:
            if task.task == Extractor.SUGGESTIONS_TASK_NAME:
                data_url = f"{SERVICE_HOST}:{SERVICE_PORT}/get_suggestions/{task.tenant}/{task.params.id}"
            else:
                data_url = None

            model_results_message = ResultsMessage(
                tenant=task.tenant,
                task=task.task,
                params=task.params,
                success=True,
                error_message="",
                data_url=data_url,
            )
        config_logger.info(model_results_message.model_dump())
        self.results_queue.sendMessage().message(model_results_message.model_dump()).execute()
        return True

    def log_process_information(self, message):
        try:
            config_logger.info(f"Processing Redis message: {message}")
            config_logger.info(
                f"Messages pending in queue: {self.task_queue.getQueueAttributes().exec_command()['msgs'] - 1}"
            )
        except redis.exceptions.ConnectionError:
            config_logger.info("No Redis messages information available")

    def subscribe_to_tasks_queue(self):
        while True:
            try:
                self.task_queue.getQueueAttributes().exec_command()
                self.results_queue.getQueueAttributes().exec_command()
                logs_queue.getQueueAttributes().exec_command()

                redis_smq_consumer = RedisSMQConsumer(
                    qname=TASK_QUEUE_NAME,
                    processor=self.process,
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                )
                redis_smq_consumer.run()
            except redis.exceptions.ConnectionError:
                config_logger.error(f"Error connecting to Redis: {REDIS_HOST}:{REDIS_PORT}")
                sleep(20)
            except cmd.exceptions.QueueDoesNotExist:
                config_logger.info("Creating queues")
                self.task_queue.createQueue().exceptions(False).execute()
                self.results_queue.createQueue().exceptions(False).execute()
                logs_queue.createQueue().exceptions(False).execute()
                config_logger.info("Queues have been created")


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

    config_logger.info("Is GPU used?")
    config_logger.info(torch.cuda.is_available())
    queue_processor = QueueProcessor()
    queue_processor.subscribe_to_tasks_queue()
