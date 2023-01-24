import re
from typing import List

from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.Method import Method
from dateparser.search import search_dates


class DateParserMethod(Method):
    @staticmethod
    def get_best_date(dates):
        not_numbers_dates = [x[1] for x in dates if re.search("[a-zA-Z]", x[0])]
        if not_numbers_dates:
            return not_numbers_dates[0]

        return dates[0][1]

    @staticmethod
    def get_date(text, languages):
        try:
            dates = search_dates(text, languages=languages)

            if not dates:
                dates = search_dates(text)

            if dates:
                return DateParserMethod.get_best_date(dates)

        except TypeError:
            return None
        except IndexError:
            return None

    def performance(self, semantic_extraction_data: List[SemanticExtractionData], training_set_length: int):
        if not semantic_extraction_data:
            return 0, []

        performance_train_set, performance_test_set = self.get_train_test(semantic_extraction_data, training_set_length)

        self.train(semantic_extraction_data)

        predictions = self.predict([x.segment_text for x in performance_test_set])
        self.log_performance_sample(semantic_extractions_data=performance_test_set, predictions=predictions)

        correct = [index for index, test in enumerate(performance_test_set) if test.text == predictions[index]]
        self.remove_model()
        return 100 * len(correct) / len(performance_test_set), predictions

    def train(self, semantic_extraction_data: List[SemanticExtractionData]):
        languages = [x.language_iso for x in semantic_extraction_data]
        self.save_json("languages.json", list(set(languages)))

    def predict(self, texts: List[str]) -> List[str]:
        languages = self.load_json("languages.json")
        predictions_dates = [self.get_date(text, languages) for text in texts]
        predictions = [date.strftime("%Y-%m-%d") if date else "" for date in predictions_dates]
        return predictions
