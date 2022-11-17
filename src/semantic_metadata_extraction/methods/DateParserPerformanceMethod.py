from typing import List

from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.Method import Method
from dateparser.search import search_dates


class DateParserPerformanceMethod(Method):
    @staticmethod
    def get_date(text, languages):
        try:
            dates = search_dates(text, languages=languages)

            if not dates:
                dates = search_dates(text)

            if dates:
                return dates[0][1]

        except TypeError:
            return None
        except IndexError:
            return None

    def performance(self, semantic_extraction_data: List[SemanticExtractionData], training_set_length: int):
        if not semantic_extraction_data:
            return 0

        performance_train_set, performance_test_set = self.get_train_test(semantic_extraction_data, training_set_length)

        self.train(performance_train_set)
        predictions = self.predict([x.segment_text for x in performance_test_set])

        correct = [index for index, test in enumerate(performance_test_set) if test.text == predictions[index]]
        self.remove_model()
        return 100 * len(correct) / len(performance_test_set), predictions

    def train(self, semantic_extraction_data: List[SemanticExtractionData]):
        languages = [x.language_iso for x in semantic_extraction_data]
        self.save_json("languages.json", languages)

    def predict(self, texts: List[str]) -> List[str]:
        languages = self.load_json("languages.json")
        predictions_dates = [self.get_date(text, languages) for text in texts]

        return [date.strftime("%d %b, %Y") if date else "" for date in predictions_dates]
