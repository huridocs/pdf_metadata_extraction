from collections import Counter
from rapidfuzz import fuzz

from data.Option import Option
from metadata_extraction.PdfDataSegment import PdfDataSegment
from multi_option_extraction.MultiOptionExtractionMethod import MultiOptionExtractionMethod
from multi_option_extraction.data.MultiOptionData import MultiOptionData


class FuzzyFirst(MultiOptionExtractionMethod):
    @staticmethod
    def get_first_appearance(pdf_segments: list[PdfDataSegment], options: list[str]) -> list[str]:
        for pdf_segment in pdf_segments:
            for ratio_threshold in range(100, 69, -10):
                for option in options:
                    if fuzz.partial_ratio(option, pdf_segment.text_content.lower()) >= ratio_threshold:
                        return [option]

        return []

    def predict(self, multi_option_data: MultiOptionData) -> list[list[Option]]:
        predictions = list()
        options_labels = [x.label.lower() for x in multi_option_data.options]
        for multi_option_sample in multi_option_data.samples:
            pdf_segments: list[PdfDataSegment] = [x for x in multi_option_sample.pdf_data.pdf_data_segments]
            prediction = self.get_first_appearance(pdf_segments, options_labels)
            if prediction:
                predictions.append([multi_option_data.options[options_labels.index(prediction[0])]])
            else:
                predictions.append([])

        return predictions

    def train(self, multi_option_data: MultiOptionData):
        pass
