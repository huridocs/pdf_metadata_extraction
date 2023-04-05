from dataclasses import dataclass

from data.Option import Option
from data.PdfTagData import PdfTagData
from data.SemanticPredictionData import SemanticPredictionData


@dataclass
class MultiOptionExtractionSample:
    pdf_tags: list[PdfTagData]
    options: list[Option]
    language_iso: str = ""


@dataclass
class MultiOptionExtractionData:
    multi_value: bool
    options: list[Option]
    samples: list[MultiOptionExtractionSample]

    def to_semantic_prediction_data(self) -> list[SemanticPredictionData]:
        return [SemanticPredictionData(pdf_tags=sample.pdf_tags) for sample in self.samples]
