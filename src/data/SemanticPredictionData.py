from dataclasses import dataclass

from data.PdfTagData import PdfTagData
from metadata_extraction.PdfDataSegment import PdfDataSegment


@dataclass
class SemanticPredictionData:
    pdf_tags_data: list[PdfTagData]

    def get_text(self):
        return " ".join([x.text for x in self.pdf_tags_data])

    @staticmethod
    def from_pdf_data_segments(pdf_data_segments: list[PdfDataSegment]):
        return SemanticPredictionData(pdf_tags_data=[PdfTagData.from_text(x.text_content) for x in pdf_data_segments])

    @staticmethod
    def from_text(text: str):
        return SemanticPredictionData(pdf_tags_data=[PdfTagData.from_text(text)])

    @staticmethod
    def from_texts(texts: list[str]):
        return [SemanticPredictionData.from_text(text) for text in texts]
