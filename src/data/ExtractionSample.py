from dataclasses import dataclass

from data.LabeledData import LabeledData
from data.PdfData import PdfData


@dataclass
class ExtractionSample:
    pdf_data: PdfData
    labeled_data: LabeledData = None

    def get_text(self):
        texts = list()
        for pdf_metadata_segment in self.pdf_data.pdf_data_segments:
            texts.append(pdf_metadata_segment.text_content)

        return " ".join(texts)
