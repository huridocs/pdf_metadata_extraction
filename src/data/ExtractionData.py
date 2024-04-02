from dataclasses import dataclass

from data.ExtractionIdentifier import ExtractionIdentifier
from data.Option import Option
from data.ExtractionSample import ExtractionSample


@dataclass
class ExtractionData:
    samples: list[ExtractionSample]
    options: list[Option]
    multi_value: bool
    extraction_identifier: ExtractionIdentifier = None
