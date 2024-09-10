from data.ExtractionData import ExtractionData
from data.PredictionSample import PredictionSample
from extractors.ToTextExtractorMethod import ToTextExtractorMethod


class SameInputOutputMethod(ToTextExtractorMethod):

    def train(self, extraction_data: ExtractionData):
        pass

    @staticmethod
    def trim_text(tag_texts: list[str]) -> str:
        text = " ".join(tag_texts)
        return " ".join(text.split())

    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        return [self.trim_text(x.tags_texts) for x in predictions_samples]
