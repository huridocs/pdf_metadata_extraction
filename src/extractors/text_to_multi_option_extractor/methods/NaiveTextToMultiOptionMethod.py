from data.Option import Option
from data.ExtractionData import ExtractionData
from data.PredictionSample import PredictionSample
from extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import TextToMultiOptionMethod


class NaiveTextToMultiOptionMethod(TextToMultiOptionMethod):

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        return True

    def predict(self, predictions_samples: list[PredictionSample]) -> list[list[Option]]:
        return [[] for _ in predictions_samples]

    def train(self, multi_option_data: ExtractionData):
        pass
