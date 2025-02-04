from data.Params import Params
from data.TaskType import TaskType


class TrainableEntityExtractionTask(TaskType):
    tenant: str
    params: Params
