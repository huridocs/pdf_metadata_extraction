from dataclasses import dataclass

from data.Option import Option
from data.PdfTagData import PdfTagData
from data.SemanticPredictionData import SemanticPredictionData


@dataclass
class MultiOptionExtractionSample:
    pdf_tags: list[PdfTagData]
    values: list[Option]
    language_iso: str = ""

    def get_text(self):
        return ' '.join([x.text for x in self.pdf_tags])


@dataclass
class MultiOptionExtractionData:
    multi_value: bool
    options: list[Option]
    samples: list[MultiOptionExtractionSample]

    def to_semantic_prediction_data(self) -> list[SemanticPredictionData]:
        return [SemanticPredictionData(pdf_tags=sample.pdf_tags) for sample in self.samples]
