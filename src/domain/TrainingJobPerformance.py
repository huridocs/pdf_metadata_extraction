from pydantic import BaseModel
from trainable_entity_extractor.domain.Performance import Performance
from domain.DistributedSubJob import DistributedSubJob


class TrainingJobPerformance(BaseModel):
    sub_job: DistributedSubJob
    performance_result: Performance
    
    @property
    def performance_score(self) -> float:
        return self.performance_result.performance
    
    @property
    def is_perfect(self) -> bool:
        return self.performance_score == 100.0
