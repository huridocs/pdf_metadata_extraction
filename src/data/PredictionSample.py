from dataclasses import dataclass

from data.PdfData import PdfData


@dataclass
class PredictionSample:
    pdf_data: PdfData = None
    tags_texts: list[str] = None
    entity_name: str = ""

    def get_text(self):
        texts = list()
        for segment in self.pdf_data.pdf_data_segments:
            texts.append(segment.text_content)

        return " ".join(texts)

    @staticmethod
    def from_pdf_data(pdf_data: PdfData):
        return PredictionSample(pdf_data=pdf_data)

    @staticmethod
    def from_text(text: str, entity_name: str = ""):
        return PredictionSample(tags_texts=[text], entity_name=entity_name)

    @staticmethod
    def from_texts(texts: list[str]):
        return PredictionSample(tags_texts=texts)
