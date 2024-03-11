from paragraph_extraction_trainer.PdfSegment import PdfSegment
from rapidfuzz import fuzz

from multi_option_extraction.PdfLabels import PdfLabels
from multi_option_extraction.PdfMultiOptionExtractionMethod import MultiOptionExtractionMethod


class FuzzyLastLabelWord(MultiOptionExtractionMethod):
    def get_first_appearance(self, pdf_segments: list[PdfSegment]) -> list[str]:
        for pdf_segment in pdf_segments:
            for ratio_threshold in range(100, 60, -10):
                for option in self.options:
                    if fuzz.partial_ratio(option.split()[-1].lower(), pdf_segment.text_content.lower()) >= ratio_threshold:
                        return [option]

        return []

    def predict(self, multi_option_samples: list[PdfLabels]):
        predictions = list()
        for pdf_label in multi_option_samples:
            pdf_segments = [PdfSegment.from_pdf_tokens(x.tokens) for x in pdf_label.paragraphs]
            predictions.append(self.get_first_appearance(pdf_segments))

        return predictions

    def train(self, multi_option_samples: list[PdfLabels]):
        pass
