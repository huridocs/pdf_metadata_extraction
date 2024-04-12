import math
from collections import Counter
from rapidfuzz import fuzz

from data.Option import Option
from data.PdfDataSegment import PdfDataSegment
from extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import PdfMultiOptionMethod

from data.ExtractionData import ExtractionData


class FuzzyFirstCleanLabel(PdfMultiOptionMethod):
    @staticmethod
    def get_first_appearance(pdf_segments: list[PdfDataSegment], options: list[str]) -> list[str]:
        for pdf_segment in pdf_segments:
            for ratio_threshold in range(100, 95, -1):
                for option in options:
                    if len(pdf_segment.text_content) < math.ceil(len(option) * ratio_threshold / 100):
                        continue
                    if fuzz.partial_ratio(option, pdf_segment.text_content.lower()) >= ratio_threshold:
                        pdf_segment.ml_label = 1
                        return [option]

        return []

    def predict(self, multi_option_data: ExtractionData) -> list[list[Option]]:
        predictions = list()
        clean_options = self.get_cleaned_options(multi_option_data.options)
        for multi_option_sample in multi_option_data.samples:
            pdf_segments: list[PdfDataSegment] = [x for x in multi_option_sample.pdf_data.pdf_data_segments]
            prediction = self.get_first_appearance(pdf_segments, clean_options)
            if prediction:
                predictions.append([multi_option_data.options[clean_options.index(prediction[0])]])
            else:
                predictions.append([])

        return predictions

    def train(self, multi_option_data: ExtractionData):
        pass

    @staticmethod
    def get_cleaned_options(options: list[Option]) -> list[str]:
        options_labels = [x.label.lower() for x in options]
        words_counter = Counter()
        for option in options_labels:
            words_counter.update(option.split())

        clean_options = list()
        for option in options_labels:
            clean_options.append(option)
            for word, count in words_counter.most_common():
                if count == 1:
                    continue

                if word not in option:
                    continue

                if clean_options[-1].replace(word, "").strip() != "":
                    clean_options[-1] = clean_options[-1].replace(word, "").strip()

        return clean_options
