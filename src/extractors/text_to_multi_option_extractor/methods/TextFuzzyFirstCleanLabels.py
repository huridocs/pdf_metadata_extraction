import math
from collections import Counter

from rapidfuzz import fuzz

from data.Option import Option
from data.ExtractionData import ExtractionData
from data.PredictionSample import PredictionSample
from extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import TextToMultiOptionMethod


class TextFuzzyFirstCleanLabels(TextToMultiOptionMethod):

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        return True

    @staticmethod
    def get_appearance(texts: list[str], options: list[str]) -> list[str]:
        all_text = " ".join(texts).lower()
        max_words = max([len(option.split()) for option in options])
        words = all_text.split()
        window_texts = [" ".join(words[i : i + max_words]) for i in range(len(words) - max_words + 1)]
        for text in window_texts:
            for ratio_threshold in range(100, 69, -10):
                for option in options:
                    if len(text) < math.ceil(len(option) * ratio_threshold / 100):
                        continue

                    if fuzz.partial_ratio(option, text.lower()) >= ratio_threshold:
                        return [option]

        return []

    def predict(self, predictions_samples: list[PredictionSample]) -> list[list[Option]]:
        predictions: list[list[Option]] = list()
        option_labels = self.get_cleaned_labels(self.options)
        for sample in predictions_samples:
            values = self.get_appearance(sample.tags_texts, option_labels)

            if values:
                predictions.append([self.options[option_labels.index(values[0])]])
            else:
                predictions.append([])

        return predictions

    def train(self, multi_option_data: ExtractionData):
        pass

    @staticmethod
    def get_cleaned_labels(options: list[Option]) -> list[str]:
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
