import os
import yaml
from rsmq.consumer import RedisSMQConsumer
from rsmq import RedisSMQ

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

        self.task_queue = self.create_queue(f'{self.SERVICE_NAME}_tasks')
        self.results_queue = self.create_queue(f'{self.SERVICE_NAME}_results')

    def set_redis_parameters_from_yml(self):
        if os.path.exists(f'{self.docker_volume_path}/redis_server.yml'):
            self.redis_host = yaml.safe_load(open(f'{self.docker_volume_path}/redis_server.yml', 'r'))['host']
            self.redis_port = int(yaml.safe_load(open(f'{self.docker_volume_path}/redis_server.yml', 'r'))['port'])

    def create_queue(self, queue_name: str):
        queue = RedisSMQ(host=self.redis_host, port=self.redis_port, qname=queue_name)
        queue.createQueue().exceptions(False).execute()
        return queue

    def execute_task(self, id, message, rc, ts):
        task = InformationExtractionTask(**message)
        task_calculated, error_message = InformationExtraction.calculate_task(task)

        if not task_calculated:
            model_results_message = ResultsMessage(tenant=task.tenant,
                                                   task=task.task,
                                                   data=task.data,
                                                   success=False,
                                                   error_message=error_message)

            self.logger.error(model_results_message.json())
        else:
            model_results_message = ResultsMessage(tenant=task.tenant,
                                                   task=task.task,
                                                   data=task.data,
                                                   success=True,
                                                   error_message='')

        self.results_queue.sendMessage().message(model_results_message.dict()).execute()

        return True

    def subscribe_to_tasks_queue(self):
        queue_name = f'{self.SERVICE_NAME}_tasks'
        extractions_tasks_queue = RedisSMQ(host=self.redis_host, port=self.redis_port,
                                           qname=queue_name)
        extractions_tasks_queue.createQueue().exceptions(False).execute()

        redis_smq_consumer = RedisSMQConsumer(qname=queue_name,
                                              processor=self.execute_task,
                                              host=self.redis_host,
                                              port=self.redis_port)
        redis_smq_consumer.run()


if __name__ == "__main__":
    queue_processor = QueueProcessor()
    queue_processor.subscribe_to_tasks_queue()
