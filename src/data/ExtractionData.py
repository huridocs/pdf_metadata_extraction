from dataclasses import dataclass

from data.ExtractionIdentifier import ExtractionIdentifier
from data.Option import Option
from data.TrainingSample import TrainingSample


@dataclass
class ExtractionData:
    samples: list[TrainingSample]
    options: list[Option]
    multi_value: bool
    extraction_identifier: ExtractionIdentifier = None
