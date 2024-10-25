from data.ExtractionData import ExtractionData
from data.PredictionSample import PredictionSample
from extractors.ToTextExtractorMethod import ToTextExtractorMethod
from extractors.GlinerDateExtractor import GlinerDateExtractor


class GlinerDateParserMethod(ToTextExtractorMethod):

    @staticmethod
    def get_date(tags_texts: list[str]):
        text = " ".join(tags_texts)
        try:
            gliner_date_extractor = GlinerDateExtractor()
            dates = gliner_date_extractor.extract_dates(text)
            return dates[0]
        except:
            pass

        return None

    def train(self, extraction_data: ExtractionData):
        pass

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        predictions_dates = [self.get_date(prediction_sample.tags_texts) for prediction_sample in predictions_samples]
        predictions = [date.strftime("%Y-%m-%d") if date else "" for date in predictions_dates]
        return predictions
