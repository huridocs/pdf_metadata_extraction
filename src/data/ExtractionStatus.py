from enum import Enum


class ExtractionStatus(Enum):
    NO_MODEL = 0
    TRAINING = 1
    TRAINED = 2
    CLOUD_MODEL = 3
