import re

from data.PdfTagData import PdfTagData
from data.SemanticExtractionData import SemanticExtractionData
from data.SemanticPredictionData import SemanticPredictionData
from extractors.text_to_text_extractor.TextToTextMethod import TextToTextMethod
from dateparser.search import search_dates


class DateParserMethod(TextToTextMethod):
    @staticmethod
    def get_best_date(dates):
        if not dates:
            return None

        not_numbers_dates = [date for date in dates if re.search("[a-zA-Z]", date[0])]
        if not_numbers_dates:
            return not_numbers_dates[0][1]

        return dates[0][1]

    @staticmethod
    def get_date(pdf_tags: list[PdfTagData], languages):
        text = TextToTextMethod.get_text_from_pdf_tags(pdf_tags)
        try:
            dates = search_dates(text, languages=languages)

            if not dates:
                dates = search_dates(text)

            return DateParserMethod.get_best_date(dates)

        except TypeError:
            return None
        except IndexError:
            return None

    def performance(self, semantic_extraction_data: list[SemanticExtractionData], training_set_length: int):
        if not semantic_extraction_data:
            return 0, []

        performance_train_set, performance_test_set = self.get_train_test(semantic_extraction_data, training_set_length)

        self.train(semantic_extraction_data)

        predictions = self.predict([x.to_semantic_prediction() for x in performance_test_set])
        self.log_performance_sample(semantic_extractions_data=performance_test_set, predictions=predictions)

        correct = [index for index, test in enumerate(performance_test_set) if test.text == predictions[index]]
        self.remove_model()
        return 100 * len(correct) / len(performance_test_set), predictions

    def train(self, semantic_extraction_data: list[SemanticExtractionData]):
        languages = [x.language_iso for x in semantic_extraction_data]
        self.save_json("languages.json", list(set(languages)))

    def predict(self, semantic_predictions_data: list[SemanticPredictionData]) -> list[str]:
        languages = self.load_json("languages.json")
        predictions_dates = [
            self.get_date(semantic_prediction_data.pdf_tags_data, languages)
            for semantic_prediction_data in semantic_predictions_data
        ]
        predictions = [date.strftime("%Y-%m-%d") if date else "" for date in predictions_dates]
        return predictions
