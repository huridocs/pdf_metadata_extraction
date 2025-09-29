from enum import Enum


class JobStatus(Enum):
    """Enumeration for job statuses"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELLED = "CANCELLED"
    RETRY = "RETRY"


class JobType(Enum):
    """Enumeration for job types"""

    TRAIN = "TRAIN"
    PREDICT = "PREDICT"
    PERFORMANCE = "PERFORMANCE"
