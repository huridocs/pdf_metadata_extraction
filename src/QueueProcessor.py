from time import sleep

import redis
from pydantic import ValidationError
from rsmq.consumer import RedisSMQConsumer
from rsmq import RedisSMQ

from ServiceConfig import ServiceConfig
from data.MetadataExtractionTask import MetadataExtractionTask
from data.ResultsMessage import ResultsMessage
from metadata_extraction.MetadataExtraction import MetadataExtraction


class QueueProcessor:
    def __init__(self):
        self.config = ServiceConfig()
        self.logger = self.config.get_logger("redis_tasks")
        self.logger.info("RedisTasksProcessor")

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
            self.logger.error(f"Not a valid message: {message}")
            return True

        self.logger.info(f"Valid message: {message}")
        task_calculated, error_message = MetadataExtraction.calculate_task(task)
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

        self.results_queue.sendMessage().message(model_results_message.dict()).execute()
        return True

    def subscribe_to_tasks_queue(self):
        while True:
            try:
                self.logger.info("Creating queues")
                self.task_queue.createQueue().exceptions(False).execute()
                self.results_queue.createQueue().exceptions(False).execute()
                self.logger.info("Queues have been created")

                redis_smq_consumer = RedisSMQConsumer(
                    qname=self.config.tasks_queue_name,
                    processor=self.process,
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                )
                redis_smq_consumer.run()
            except redis.exceptions.ConnectionError:
                self.logger.error(f"Error connecting to redis: {self.config.redis_host}:{self.config.redis_port}")
                sleep(20)


if __name__ == "__main__":
    queue_processor = QueueProcessor()
    queue_processor.subscribe_to_tasks_queue()
