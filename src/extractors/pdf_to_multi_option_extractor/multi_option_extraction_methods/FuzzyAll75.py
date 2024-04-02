import math

from rapidfuzz import fuzz

from data.Option import Option
from data.PdfDataSegment import PdfDataSegment
from extractors.pdf_to_multi_option_extractor.MultiOptionExtractionMethod import MultiOptionExtractionMethod
from data.ExtractionData import ExtractionData

threshold = 75


class FuzzyAll75(MultiOptionExtractionMethod):
    @staticmethod
    def get_appearances(pdf_segments: list[PdfDataSegment], options: list[str]) -> list[str]:
        appearances = []
        for pdf_segment in pdf_segments:
            for option in options:
                if len(pdf_segment.text_content) < math.ceil(len(option) * 0.85):
                    continue

                if fuzz.partial_ratio(option, pdf_segment.text_content.lower()) >= threshold:
                    pdf_segment.ml_label = 1
                    appearances.append(option)

        return list(set(appearances))

    def predict(self, multi_option_data: ExtractionData) -> list[list[Option]]:
        predictions = list()
        options_labels = [x.label.lower() for x in multi_option_data.options]
        for multi_option_sample in multi_option_data.samples:
            pdf_segments: list[PdfDataSegment] = [x for x in multi_option_sample.pdf_data.pdf_data_segments]
            predictions_sample = self.get_appearances(pdf_segments, options_labels)
            predictions.append([multi_option_data.options[options_labels.index(x)] for x in predictions_sample])

        return predictions

    def train(self, multi_option_data: ExtractionData):
        pass