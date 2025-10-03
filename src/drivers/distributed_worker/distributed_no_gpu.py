from celery import Celery
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob

from config import REDIS_HOST, REDIS_PORT, NAME
from drivers.distributed_worker.distributed_flow import (
    distributed_predict,
    train_one_method,
    performance_one_method,
)


app = Celery(NAME, broker=f"redis://{REDIS_HOST}:{REDIS_PORT}", backend=f"redis://{REDIS_HOST}:{REDIS_PORT}")


@app.task
def predict_no_gpu(extractor_job_dict: dict) -> bool:
    extractor_job = TrainableEntityExtractorJob.model_validate(extractor_job_dict)
    return distributed_predict(extractor_job)


@app.task
def train_no_gpu(extractor_job_dict: dict) -> bool:
    extractor_job = TrainableEntityExtractorJob.model_validate(extractor_job_dict)
    return train_one_method(extractor_job)


@app.task
def performance_no_gpu(extractor_job_dict: dict) -> dict:
    extractor_job = TrainableEntityExtractorJob.model_validate(extractor_job_dict)
    return performance_one_method(extractor_job).model_dump()
