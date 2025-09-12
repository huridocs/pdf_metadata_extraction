from enum import StrEnum


class DistributedJobType(StrEnum):
    PERFORMANCE = "PERFORMANCE"
    TRAIN = "TRAIN"
    PREDICT = "PREDICT"

