import os
from time import sleep

import redis
import yaml
from pydantic import ValidationError
from rsmq.consumer import RedisSMQConsumer
from rsmq import RedisSMQ

from config_creator import get_server_port, get_redis_port
from data.InformationExtractionTask import InformationExtractionTask
from data.ResultsMessage import ResultsMessage
from get_logger import get_logger
from information_extraction.InformationExtraction import InformationExtraction


class QueueProcessor:
    SERVICE_NAME = 'information_extraction'

    def __init__(self):
        self.docker_volume_path = f'{os.path.dirname(os.path.realpath(__file__))}/docker_volume'

        self.logger = get_logger('redis_tasks')
        self.logger.info('RedisTasksProcessor')

        self.redis_host = 'redis_information_extraction'
        self.redis_port = 6379
        self.set_redis_parameters_from_yml()

        self.service_url = f'http://localhost:{get_server_port()}'
        self.set_server_parameters_from_yml()

        self.task_queue = RedisSMQ(host=self.redis_host, port=self.redis_port, qname=f'{self.SERVICE_NAME}_tasks')
        self.results_queue = RedisSMQ(host=self.redis_host, port=self.redis_port, qname=f'{self.SERVICE_NAME}_results')

    def set_redis_parameters_from_yml(self):
        if not os.path.exists(f'config.yml'):
            return

        with open(f'config.yml', 'r') as f:
            config_dict = yaml.safe_load(f)
            if not config_dict:
                return

            self.redis_host = config_dict['redis_host'] if 'redis_host' in config_dict else self.redis_host
            self.redis_port = int(config_dict['redis_port']) if 'redis_port' in config_dict else self.redis_port

    def set_server_parameters_from_yml(self):
        if not os.path.exists(f'config.yml'):
            return

        with open(f'config.yml', 'r') as f:
            config_dict = yaml.safe_load(f)
            if not config_dict:
                return

            service_host = config_dict['service_host'] if 'service_host' in config_dict else 'localhost'
            service_port = int(config_dict['service_port']) if 'service_port' in config_dict else get_server_port()
            self.service_url = f'http://{service_host}:{service_port}'

    def process(self, id, message, rc, ts):
        try:
            task = InformationExtractionTask(**message)
        except ValidationError:
            self.logger.error(f'Not a valid message: {message}')
            return True

        self.logger.error(f'Valid message: {message}')
        task_calculated, error_message = InformationExtraction.calculate_task(task)

        self.logger.error(f'Error: {error_message}')

        if not task_calculated:
            self.logger.error(f'Task not ok')

            model_results_message = ResultsMessage(tenant=task.tenant,
                                                   task=task.task,
                                                   params=task.params,
                                                   success=False,
                                                   error_message=error_message)

            self.logger.error(model_results_message.json())
        else:
            self.logger.error(f'Task ok')

            if task.task == InformationExtraction.SUGGESTIONS_TASK_NAME:
                data_url = f"{self.service_url}/get_suggestions/{task.tenant}/{task.params.property_name}"
            else:
                data_url = None

            model_results_message = ResultsMessage(tenant=task.tenant,
                                                   task=task.task,
                                                   params=task.params,
                                                   success=True,
                                                   error_message='',
                                                   data_url=data_url)

        self.logger.error(f'Sending {str(model_results_message.json())}')
        self.results_queue.sendMessage().message(model_results_message.dict()).execute()

        return True

    def subscribe_to_tasks_queue(self):
        while True:
            try:
                self.task_queue.createQueue().exceptions(False).execute()
                self.results_queue.createQueue().exceptions(False).execute()

                queue_name = f'{self.SERVICE_NAME}_tasks'

                redis_smq_consumer = RedisSMQConsumer(qname=queue_name,
                                                      processor=self.process,
                                                      host=self.redis_host,
                                                      port=self.redis_port)
                redis_smq_consumer.run()
            except redis.exceptions.ConnectionError:
                self.logger.error(f'Error connecting to redis: {self.redis_host}:{self.redis_port}')
                sleep(20)


if __name__ == "__main__":
    queue_processor = QueueProcessor()
    queue_processor.subscribe_to_tasks_queue()
