from dataclasses import dataclass
from typing import List

from data.Option import Option
from data.SemanticExtractionData import SemanticExtractionData


@dataclass
class MultiOptionExtractionSample:
    text: str
    options: List[Option]


@dataclass
class MultiOptionExtractionData:
    multilingual: bool
    multi_value: bool
    options: List[Option]
    samples: List[MultiOptionExtractionSample]

    def to_semantic_extraction_data(self) -> List[SemanticExtractionData]:
        semantic_extraction_data_list: List[SemanticExtractionData] = list()
        language_iso = "multilingual" if self.multilingual else "en"

        for sample in self.samples:
            text = " ; ".join([option.label for option in sample.options])
            semantic_extraction_data = SemanticExtractionData(text=text, segment_text=sample.text, language_iso=language_iso)
            semantic_extraction_data_list.append(semantic_extraction_data)

        return semantic_extraction_data_list
