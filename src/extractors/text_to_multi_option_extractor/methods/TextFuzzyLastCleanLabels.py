import math

from rapidfuzz import fuzz

from extractors.text_to_multi_option_extractor.methods.TextFuzzyFirstCleanLabels import TextFuzzyFirstCleanLabels


class TextFuzzyLastCleanLabels(TextFuzzyFirstCleanLabels):
    @staticmethod
    def get_appearance(texts: list[str], options: list[str]) -> list[str]:
        all_text = " ".join(texts).lower()
        max_words = max([len(option.split()) for option in options])
        words = all_text.split()
        window_texts = [" ".join(words[i : i + max_words]) for i in range(len(words) - max_words + 1)]
        for text in reversed(window_texts):
            for ratio_threshold in range(100, 69, -10):
                for option in options:
                    if len(text) < math.ceil(len(option) * ratio_threshold / 100):
                        continue

                    if fuzz.partial_ratio(option, text.lower()) >= ratio_threshold:
                        return [option]

        return []
