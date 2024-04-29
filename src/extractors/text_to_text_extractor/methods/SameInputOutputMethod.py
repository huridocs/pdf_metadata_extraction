from data.ExtractionData import ExtractionData
from data.PredictionSample import PredictionSample
from extractors.text_to_text_extractor.TextToTextMethod import TextToTextMethod


class SameInputOutputMethod(TextToTextMethod):

    def train(self, extraction_data: ExtractionData):
        pass

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        return [" ".join(x.tags_texts) for x in predictions_samples]
