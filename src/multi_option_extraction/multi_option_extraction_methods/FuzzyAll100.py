from rapidfuzz import fuzz

from data.Option import Option
from metadata_extraction.PdfDataSegment import PdfDataSegment
from multi_option_extraction.MultiOptionExtractionMethod import MultiOptionExtractionMethod
from multi_option_extraction.data.MultiOptionData import MultiOptionData

threshold = 100


class FuzzyAll100(MultiOptionExtractionMethod):
    @staticmethod
    def get_appearances(pdf_segments: list[PdfDataSegment], options: list[str]) -> list[str]:
        appearances = []
        for pdf_segment in pdf_segments:
            for option in options:
                if fuzz.partial_ratio(option, pdf_segment.text_content.lower()) >= threshold:
                    appearances.append(option)

        return list(set(appearances))

    def predict(self, multi_option_data: MultiOptionData) -> list[list[Option]]:
        predictions = list()
        options_labels = [x.label.lower() for x in multi_option_data.options]
        for multi_option_sample in multi_option_data.samples:
            pdf_segments: list[PdfDataSegment] = [x for x in multi_option_sample.pdf_data.pdf_data_segments]
            predictions_sample = self.get_appearances(pdf_segments, options_labels)
            predictions.append([multi_option_data.options[options_labels.index(x)] for x in predictions_sample])

        return predictions

    def train(self, multi_option_data: MultiOptionData):
        pass
