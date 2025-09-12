from pathlib import Path

from celery import Celery
from trainable_entity_extractor.domain.ExtractionDistributedTask import ExtractionDistributedTask
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.Performance import Performance

from config import REDIS_HOST, REDIS_PORT, NAME
from drivers.distributed_worker.distributed_worker import distributed_predict, train_one_method, performance_one_method
from drivers.distributed_worker.model_to_cloud import upload_model_to_cloud

app = Celery(NAME,
             broker=f'redis://{REDIS_HOST}:{REDIS_PORT}',
             backend=f'redis://{REDIS_HOST}:{REDIS_PORT}')

@app.task
def upload_model(extraction_identifier: ExtractionIdentifier, method_name: str):
    if not Path(extraction_identifier.get_path(), method_name).exists():
        return

    upload_model_to_cloud(extraction_identifier, extraction_identifier.run_name)

@app.task
def predict_no_gpu(extraction_distributed_task: ExtractionDistributedTask) -> tuple[bool, str]:
    return distributed_predict(extraction_distributed_task)

@app.task
def train_no_gpu(extraction_distributed_task: ExtractionDistributedTask, options: list[Option], multi_value: bool) -> tuple[bool, str]:
    return train_one_method(extraction_distributed_task, options, multi_value)

@app.task
def performance_no_gpu(extraction_distributed_task: ExtractionDistributedTask, options: list[Option], multi_value: bool) -> Performance:
    return performance_one_method(extraction_distributed_task, options, multi_value)