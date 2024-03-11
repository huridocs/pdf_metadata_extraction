from paragraph_extraction_trainer.PdfSegment import PdfSegment
from rapidfuzz import fuzz

from multi_option_extraction.PdfLabels import PdfLabels
from multi_option_extraction.PdfMultiOptionExtractionMethod import MultiOptionExtractionMethod

ratio_threshold = 100


class FuzzyAll100Method(MultiOptionExtractionMethod):
    def get_all_appearance(self, pdf_segments: list[PdfSegment]) -> list[str]:
        appearances = []
        for option in self.options:
            for pdf_segment in pdf_segments:
                ratio = fuzz.partial_ratio(option.lower(), pdf_segment.text_content.lower())
                if ratio >= ratio_threshold:
                    appearances.append(option)

        return list(set(appearances))

    def predict(self, multi_option_samples: list[PdfLabels]):
        predictions = list()
        for pdf_label in multi_option_samples:
            pdf_segments = [PdfSegment.from_pdf_tokens(x.tokens) for x in pdf_label.paragraphs]
            predictions.append(self.get_all_appearance(pdf_segments))

        return predictions

    def train(self, multi_option_samples: list[PdfLabels]):
        pass
