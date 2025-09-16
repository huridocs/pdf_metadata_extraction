import os

import torch
from celery import Celery
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.Performance import Performance

from config import REDIS_HOST, REDIS_PORT, NAME
from drivers.distributed_worker.distributed_flow import distributed_predict, train_one_method, performance_one_method

app = Celery(NAME,
             broker=f'redis://{REDIS_HOST}:{REDIS_PORT}',
             backend=f'redis://{REDIS_HOST}:{REDIS_PORT}')

@app.task
def predict_gpu(extractor_job: TrainableEntityExtractorJob) -> tuple[bool, str]:
    is_gpu_available()
    return distributed_predict(extractor_job)

@app.task
def train_gpu(extractor_job: TrainableEntityExtractorJob, options: list[Option], multi_value: bool) -> tuple[bool, str]:
    is_gpu_available()
    return train_one_method(extractor_job, options, multi_value)

@app.task
def performance_gpu(extractor_job: TrainableEntityExtractorJob, options: list[Option], multi_value: bool) -> Performance:
    is_gpu_available()
    return performance_one_method(extractor_job, options, multi_value)

def is_gpu_available():
    if not torch.cuda.is_available():
        os._exit(1)