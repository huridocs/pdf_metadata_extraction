from celery import current_app
from celery.result import AsyncResult
from pydantic import BaseModel
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob


class DistributedSubJob(BaseModel):
    job_id: str | None
    extractor_job: TrainableEntityExtractorJob
    retry_count: int = 0
    max_retries: int = 5

    @property
    def job(self) -> AsyncResult:
        return current_app.AsyncResult(self.job_id)

    def status(self) -> str:
        return self.job.status

    def result(self):
        return self.job.result
