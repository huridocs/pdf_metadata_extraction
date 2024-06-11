import re

from rapidfuzz import fuzz

from data.Option import Option
from data.PdfDataSegment import PdfDataSegment
from extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import PdfMultiOptionMethod

from data.ExtractionData import ExtractionData


class FuzzyCommas(PdfMultiOptionMethod):

    threshold = 92

    def get_appearances(self, pdf_segments: list[PdfDataSegment], options: list[str]) -> list[str]:
        appearances = []
        for pdf_segment in pdf_segments:
            text = pdf_segment.text_content.lower()
            texts_separated_by_comma = self.clean_options(re.split(",|:| and ", text))
            for one_piece_text in texts_separated_by_comma:
                for option in options:
                    if len(one_piece_text) < len(option) * 0.92 or len(one_piece_text) > len(option) * 1.2:
                        continue

                    if fuzz.partial_ratio(option, one_piece_text) >= self.threshold:
                        pdf_segment.ml_label = 1
                        appearances.append(option)
                        break

        return appearances

    @staticmethod
    def clean_option(option: str) -> str:
        option = option.lower()
        option = "".join([letter for letter in option if letter.isalnum() or letter == " "])
        option = " ".join(sorted(option.split()))
        return option

    def clean_options(self, options: list[str]) -> list[str]:
        return list([self.clean_option(option) for option in options])

    def predict(self, multi_option_data: ExtractionData) -> list[list[Option]]:
        predictions = list()
        options_labels = self.clean_options([x.label for x in multi_option_data.options])
        clean_options_sorted = list(sorted(options_labels, key=lambda x: len(x), reverse=True))

        for multi_option_sample in multi_option_data.samples:
            pdf_segments: list[PdfDataSegment] = [x for x in multi_option_sample.pdf_data.pdf_data_segments]
            predictions_sample = self.get_appearances(pdf_segments, clean_options_sorted)
            predictions.append([multi_option_data.options[options_labels.index(x)] for x in predictions_sample])

        return predictions

    def train(self, multi_option_data: ExtractionData):
        pass
