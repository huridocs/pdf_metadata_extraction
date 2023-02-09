from pydantic import BaseModel

from data.PdfTagData import PdfTagData


class SemanticPredictionData(BaseModel):
    pdf_tags: list[PdfTagData]

    @staticmethod
    def from_text(text: str):
        return SemanticPredictionData(pdf_tags=[PdfTagData.from_text(text)])

    @staticmethod
    def from_texts(texts: list[str]):
        return [SemanticPredictionData.from_text(text) for text in texts]
