import math

from rapidfuzz import fuzz

from data.Option import Option
from data.ExtractionData import ExtractionData
from data.PredictionSample import PredictionSample
from extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import TextToMultiOptionMethod


class TextFuzzyFirst(TextToMultiOptionMethod):
    @staticmethod
    def get_first_appearance(texts: list[str], options: list[str]) -> list[str]:
        for text in texts:
            for ratio_threshold in range(100, 69, -10):
                for option in options:
                    if len(text) < math.ceil(len(option) * ratio_threshold / 100):
                        continue

                    if fuzz.partial_ratio(option, text.lower()) >= ratio_threshold:
                        return [option]

        return []

    def predict(self, predictions_samples: list[PredictionSample]) -> list[list[Option]]:
        predictions: list[list[Option]] = list()
        option_labels = [option.label.lower() for option in self.options]
        for sample in predictions_samples:
            values = self.get_first_appearance(sample.tags_texts, option_labels)
            predictions.append([option for option in self.options if option.label in values])

        return predictions

    def train(self, multi_option_data: ExtractionData):
        pass
