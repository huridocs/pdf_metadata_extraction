from pydantic import BaseModel, Field
from domain.DistributedJobType import DistributedJobType
from domain.DistributedSubJob import DistributedSubJob
from domain.TrainableEntityExtractionTask import TrainableEntityExtractionTask
from datetime import datetime

class DistributedJob(BaseModel):
    type: DistributedJobType
    task: TrainableEntityExtractionTask
    sub_jobs: list[DistributedSubJob]
    start_time: datetime = Field(default_factory=datetime.now)
    queue_name: str
