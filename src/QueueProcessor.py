import os
from time import sleep

import redis
from pydantic import ValidationError
from rsmq.consumer import RedisSMQConsumer
from rsmq import RedisSMQ, cmd
from sentry_sdk.integrations.redis import RedisIntegration
import sentry_sdk

from ServiceConfig import ServiceConfig
from data.MetadataExtractionTask import MetadataExtractionTask
from data.ResultsMessage import ResultsMessage
from metadata_extraction.MetadataExtraction import MetadataExtraction


class QueueProcessor:
    def __init__(self):
        self.config = ServiceConfig()
        self.logger = self.config.get_logger("redis_tasks")
        self.logger.info("The PDF Metadata Extractor has been started")

        self.task_queue = RedisSMQ(
            host=self.config.redis_host,
            port=self.config.redis_port,
            qname=self.config.tasks_queue_name,
        )

        self.results_queue = RedisSMQ(
            host=self.config.redis_host,
            port=self.config.redis_port,
            qname=self.config.results_queue_name,
        )

    def process(self, id, message, rc, ts):
        try:
            task = MetadataExtractionTask(**message)
        except ValidationError:
            self.logger.error(f"Not a valid Redis message: {message}")
            return True

        self.log_process_information(message)

        task_calculated, error_message = MetadataExtraction.calculate_task(task, self.logger)
        if error_message:
            self.logger.info(f"Error: {error_message}")

        if not task_calculated:
            model_results_message = ResultsMessage(
                tenant=task.tenant,
                task=task.task,
                params=task.params,
                success=False,
                error_message=error_message,
            )
        else:
            data_url = None
            if task.task == MetadataExtraction.SUGGESTIONS_TASK_NAME:
                data_url = f"{self.config.service_url}/get_suggestions/{task.tenant}/{task.params.property_name}"

            model_results_message = ResultsMessage(
                tenant=task.tenant,
                task=task.task,
                params=task.params,
                success=True,
                error_message="",
                data_url=data_url,
            )

        self.logger.info(f"Results Redis message: {model_results_message}")
        self.results_queue.sendMessage().message(model_results_message.dict()).execute()
        return True

    def log_process_information(self, message):
        try:
            self.logger.info(f"Processing Redis message: {message}")
            self.logger.info(f"Messages pending in queue: {self.task_queue.getQueueAttributes().exec_command()['msgs'] - 1}")
        except redis.exceptions.ConnectionError:
            self.logger.info("No Redis messages information available")

    def subscribe_to_tasks_queue(self):
        while True:
            try:
                self.task_queue.getQueueAttributes().exec_command()
                self.results_queue.getQueueAttributes().exec_command()

                redis_smq_consumer = RedisSMQConsumer(
                    qname=self.config.tasks_queue_name,
                    processor=self.process,
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                )
                redis_smq_consumer.run()
            except redis.exceptions.ConnectionError:
                self.logger.error(f"Error connecting to Redis: {self.config.redis_host}:{self.config.redis_port}")
                sleep(20)
            except cmd.exceptions.QueueDoesNotExist:
                self.logger.info("Creating queues")
                self.task_queue.createQueue().exceptions(False).execute()
                self.results_queue.createQueue().exceptions(False).execute()
                self.logger.info("Queues have been created")


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

    queue_processor = QueueProcessor()
    queue_processor.subscribe_to_tasks_queue()
