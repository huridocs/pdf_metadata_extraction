import math

from rapidfuzz import fuzz

from data.Option import Option
from data.ExtractionData import ExtractionData
from data.PredictionSample import PredictionSample
from extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import TextToMultiOptionMethod

threshold = 75


class TextFuzzyAll75(TextToMultiOptionMethod):
    @staticmethod
    def get_appearances(texts: list[str], options: list[str]) -> list[str]:
        appearances = []
        for text in texts:
            for option in options:
                if len(text) < math.ceil(len(option) * 0.85):
                    continue

                if fuzz.partial_ratio(option, text.lower()) >= threshold:
                    appearances.append(option)

        return list(set(appearances))

    def predict(self, predictions_samples: list[PredictionSample]) -> list[list[Option]]:
        predictions: list[list[Option]] = list()
        option_labels = [option.label.lower() for option in self.options]
        for sample in predictions_samples:
            values = self.get_appearances(sample.tags_texts, option_labels)
            predictions.append([option for option in self.options if option.label in values])

        return predictions

    def train(self, multi_option_data: ExtractionData):
        pass
