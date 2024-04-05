import re

from data.ExtractionData import ExtractionData
from data.PdfTagData import PdfTagData
from data.PredictionSample import PredictionSample
from extractors.ExtractorBase import ExtractorBase
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
    def get_date(tags_texts: list[str], languages):
        text = " ".join(tags_texts)
        try:
            dates = search_dates(text, languages=languages)

            if not dates:
                dates = search_dates(text)

            return DateParserMethod.get_best_date(dates)

        except TypeError:
            return None
        except IndexError:
            return None

    def train(self, extraction_data: ExtractionData):
        languages = [x.labeled_data.language_iso for x in extraction_data.samples]
        self.save_json("languages.json", list(set(languages)))

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        languages = self.load_json("languages.json")
        predictions_dates = [
            self.get_date(prediction_sample.tags_texts, languages) for prediction_sample in predictions_samples
        ]
        predictions = [date.strftime("%Y-%m-%d") if date else "" for date in predictions_dates]
        return predictions
