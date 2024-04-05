from dataclasses import dataclass

from data.PdfData import PdfData


@dataclass
class PredictionSample:
    pdf_data: PdfData
    source_text: list[str]

    def get_text(self):
        texts = list()
        for segment in self.pdf_data.pdf_data_segments:
            texts.append(segment.text_content)

        return " ".join(texts)

    @staticmethod
    def from_pdf_data(pdf_data: PdfData):
        return PredictionSample(pdf_data=pdf_data)
