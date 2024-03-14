from dataclasses import dataclass

from data.ExtractionIdentifier import ExtractionIdentifier
from data.Option import Option
from multi_option_extraction.data.MultiOptionSample import MultiOptionSample


@dataclass
class MultiOptionData:
    samples: list[MultiOptionSample]
    options: list[Option]
    multi_value: bool
    extraction_identifier: ExtractionIdentifier = None
