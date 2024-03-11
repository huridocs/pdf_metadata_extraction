from dataclasses import dataclass

from data.Option import Option
from metadata_extraction.PdfData import PdfData


@dataclass
class MultiOptionSample:
    pdf_data: PdfData
    values: list[Option] = None
    language_iso: str = ""

    def get_text(self):
        texts = list()
        for pdf_metadata_segment in self.pdf_data.pdf_data_segments:
            texts.append(pdf_metadata_segment.text_content)

        return " ".join(texts)
