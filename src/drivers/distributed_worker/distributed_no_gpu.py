from pathlib import Path

from celery import Celery
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.Performance import Performance

from config import REDIS_HOST, REDIS_PORT, NAME
from drivers.distributed_worker.distributed_flow import distributed_predict, train_one_method, performance_one_method
from drivers.distributed_worker.model_to_cloud import upload_model_to_cloud, upload_completion_signal
from config import EXTRACTOR_JOB_PATH
from trainable_entity_extractor.use_cases.send_logs import send_logs


app = Celery(NAME, broker=f"redis://{REDIS_HOST}:{REDIS_PORT}", backend=f"redis://{REDIS_HOST}:{REDIS_PORT}")


@app.task
def upload_model(extraction_identifier: ExtractionIdentifier, method_name: str, extractor_job: TrainableEntityExtractorJob = None):
    if not Path(extraction_identifier.get_path(), method_name).exists():
        return False  # Model not found on this worker

    if extractor_job:
        job_path = Path(extraction_identifier.get_path(), EXTRACTOR_JOB_PATH)
        job_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(job_path, "w", encoding="utf-8") as file:
                file.write(extractor_job.model_dump_json())
            send_logs(extraction_identifier, f"Extractor job saved successfully to {job_path}")
        except Exception as e:
            send_logs(extraction_identifier, f"Error saving extractor job: {e}")
            return False

    # Upload model to cloud
    upload_success = upload_model_to_cloud(extraction_identifier, extraction_identifier.run_name)
    if upload_success:
        # Upload completion signal to indicate model is fully uploaded
        upload_completion_signal(extraction_identifier, extraction_identifier.run_name)

    return upload_success


@app.task
def predict_no_gpu(extractor_job: TrainableEntityExtractorJob) -> tuple[bool, str]:
    return distributed_predict(extractor_job)


@app.task
def train_no_gpu(extractor_job: TrainableEntityExtractorJob, options: list[Option], multi_value: bool) -> tuple[bool, str]:
    return train_one_method(extractor_job, options, multi_value)


@app.task
def performance_no_gpu(extractor_job: TrainableEntityExtractorJob, options: list[Option], multi_value: bool) -> Performance:
    return performance_one_method(extractor_job, options, multi_value)
