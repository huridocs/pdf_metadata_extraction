from celery import Celery
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier

from config import REDIS_HOST, REDIS_PORT, NAME
from drivers.distributed_worker.distributed_flow import (
    distributed_predict,
    train_one_method,
    performance_one_method,
    cloud_storage,
)


app = Celery(NAME, broker=f"redis://{REDIS_HOST}:{REDIS_PORT}", backend=f"redis://{REDIS_HOST}:{REDIS_PORT}")


@app.task
def upload_model(extraction_identifier_dict: dict, extractor_job_dict: dict) -> bool:
    try:
        extraction_identifier = ExtractionIdentifier.model_validate(extraction_identifier_dict)
        extractor_job = TrainableEntityExtractorJob.model_validate(extractor_job_dict)
        success = cloud_storage.upload_model(extraction_identifier, extractor_job)
        if success:
            cloud_storage.create_model_completion_signal(extraction_identifier)
            return True
        else:
            return False
    except Exception:
        return False


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
