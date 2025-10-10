import os
import time

import torch
from celery import Celery
from celery.schedules import crontab
from trainable_entity_extractor.config import config_logger

from drivers.distributed_worker.distributed_no_gpu import predict_no_gpu, train_no_gpu, performance_no_gpu
from config import REDIS_HOST, REDIS_PORT, NAME


app = Celery(NAME, broker=f"redis://{REDIS_HOST}:{REDIS_PORT}", backend=f"redis://{REDIS_HOST}:{REDIS_PORT}")

WORKER_START_TIME = time.time()
GRACE_PERIOD_SECONDS = 60 * 30

app.conf.beat_schedule = {
    "health-check-every-3-hours": {
        "task": "drivers.distributed_worker.distributed_gpu.scheduled_health_check",
        "schedule": crontab(minute="0", hour="*/3"),
        "options": {
            "expires": 30,
        },
    },
}


@app.task
def scheduled_health_check():
    config_logger.info("Scheduled health check executed.")

    for i in range(5):
        if not torch.cuda.is_available():
            config_logger.warning(f"CUDA is not available. Attempt {i + 1}/5")
            time.sleep(5)

    if not torch.cuda.is_available():
        config_logger.error("CUDA is not available after 5 attempts. Exiting.")
        os._exit(1)

    uptime = time.time() - WORKER_START_TIME
    if uptime < GRACE_PERIOD_SECONDS:
        return

    if not has_pending_tasks():
        config_logger.info("No pending tasks found. Exiting worker.")
        os._exit(1)
    else:
        config_logger.info("Pending tasks found. Continuing worker operation.")


def filter_health_check(tasks_dict):
    if not tasks_dict:
        return False
    for worker_tasks in tasks_dict.values():
        if not worker_tasks:
            continue
        filtered_tasks = [task for task in worker_tasks if "scheduled" not in task.get("name")]
        if filtered_tasks:
            return True
    return False


def has_pending_tasks():
    inspect = app.control.inspect()

    active_tasks = inspect.active()
    scheduled_tasks = inspect.scheduled()
    reserved_tasks = inspect.reserved()

    if filter_health_check(active_tasks):
        return True
    if filter_health_check(scheduled_tasks):
        return True
    if filter_health_check(reserved_tasks):
        return True

    return False


@app.task
def predict_gpu(extractor_job_dict: dict) -> bool:
    return predict_no_gpu(extractor_job_dict)


@app.task
def train_gpu(extractor_job_dict: dict) -> bool:
    return train_no_gpu(extractor_job_dict)


@app.task
def performance_gpu(extractor_job_dict: dict) -> dict:
    return performance_no_gpu(extractor_job_dict)
