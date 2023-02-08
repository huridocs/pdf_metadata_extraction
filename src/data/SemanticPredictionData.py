from pydantic import BaseModel

from data.PdfTagData import PdfTagData


class SemanticPredictionData(BaseModel):
    pdf_tags: list[PdfTagData]
    xml_file_name: str = ""

    @staticmethod
    def from_text(text: str):
        return SemanticPredictionData(pdf_tags=[PdfTagData.from_text(text)])
