from pydantic import BaseModel

from data.PdfTagData import PdfTagData
from data.SemanticPredictionData import SemanticPredictionData


class SemanticExtractionData(BaseModel):
    text: str
    pdf_tags: list[PdfTagData]
    language_iso: str

    def to_semantic_prediction(self):
        return SemanticPredictionData(pdf_tags_data=self.pdf_tags)
