import os

import torch
from celery import Celery
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob

from drivers.distributed_worker.distributed_no_gpu import predict_no_gpu, train_no_gpu, performance_no_gpu
from config import REDIS_HOST, REDIS_PORT, NAME
from drivers.distributed_worker.distributed_flow import train_one_method, performance_one_method

app = Celery(NAME, broker=f"redis://{REDIS_HOST}:{REDIS_PORT}", backend=f"redis://{REDIS_HOST}:{REDIS_PORT}")


@app.task
def predict_gpu(extractor_job_dict: dict) -> bool:
    is_gpu_available()
    return predict_no_gpu(extractor_job_dict)


@app.task
def train_gpu(extractor_job_dict: dict) -> bool:
    is_gpu_available()
    return train_no_gpu(extractor_job_dict)


@app.task
def performance_gpu(extractor_job_dict: dict) -> dict:
    is_gpu_available()
    return performance_no_gpu(extractor_job_dict)


def is_gpu_available():
    if not torch.cuda.is_available():
        os._exit(1)
