from paragraph_extraction_trainer.PdfSegment import PdfSegment
from rapidfuzz import fuzz

from multi_option_extraction.PdfLabels import PdfLabels
from multi_option_extraction.PdfMultiOptionExtractionMethod import PdfMultiOptionExtractionMethod


class FuzzyLastMethod(PdfMultiOptionExtractionMethod):
    def get_last_appearance(self, pdf_segments: list[PdfSegment]) -> list[str]:
        for pdf_segment in reversed(pdf_segments):
            for ratio_threshold in range(100, 60, -10):
                for option in self.options:
                    if fuzz.partial_ratio(option.lower(), pdf_segment.text_content.lower()) >= ratio_threshold:
                        return [option]

        return []

    def predict(self, pdfs_labels: list[PdfLabels]):
        predictions = list()
        for pdf_label in pdfs_labels:
            pdf_segments = [PdfSegment.from_pdf_tokens(x.tokens) for x in pdf_label.paragraphs]
            predictions.append(self.get_last_appearance(pdf_segments))

        return predictions

    def train(self, pdfs_labels: list[PdfLabels]):
        pass
