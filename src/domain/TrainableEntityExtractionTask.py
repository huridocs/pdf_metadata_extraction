from domain.Params import Params
from domain.TaskType import TaskType


class TrainableEntityExtractionTask(TaskType):
    tenant: str
    params: Params
