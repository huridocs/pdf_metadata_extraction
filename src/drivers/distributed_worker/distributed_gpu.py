import os

import torch
from celery import Celery
from trainable_entity_extractor.domain.ExtractionDistributedTask import ExtractionDistributedTask
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.Performance import Performance

from config import REDIS_HOST, REDIS_PORT, NAME
from drivers.distributed_worker.distributed_worker import distributed_predict, train_one_method, performance_one_method

app = Celery(NAME,
             broker=f'redis://{REDIS_HOST}:{REDIS_PORT}',
             backend=f'redis://{REDIS_HOST}:{REDIS_PORT}')

@app.task
def predict_gpu(extraction_distributed_task: ExtractionDistributedTask) -> tuple[bool, str]:
    is_gpu_available()
    return distributed_predict(extraction_distributed_task)

@app.task
def train_gpu(extraction_distributed_task: ExtractionDistributedTask, options: list[Option], multi_value: bool) -> tuple[bool, str]:
    is_gpu_available()
    return train_one_method(extraction_distributed_task, options, multi_value)

@app.task
def performance_gpu(extraction_distributed_task: ExtractionDistributedTask, options: list[Option], multi_value: bool) -> Performance:
    is_gpu_available()
    return performance_one_method(extraction_distributed_task, options, multi_value)

def is_gpu_available():
    if not torch.cuda.is_available():
        os._exit(1)