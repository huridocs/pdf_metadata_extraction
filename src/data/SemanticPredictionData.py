from pydantic import BaseModel

from data.PdfTagData import PdfTagData


class SemanticPredictionData(BaseModel):
    pdf_tags: list[PdfTagData]

    def get_text(self):
        return " ".join([x.text for x in self.pdf_tags])

    @staticmethod
    def from_text(text: str):
        return SemanticPredictionData(pdf_tags=[PdfTagData.from_text(text)])

    @staticmethod
    def from_texts(texts: list[str]):
        return [SemanticPredictionData.from_text(text) for text in texts]
