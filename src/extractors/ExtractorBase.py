from abc import abstractmethod

from data.ExtractionData import ExtractionData
from data.ExtractionIdentifier import ExtractionIdentifier
from data.PredictionSample import PredictionSample
from data.Suggestion import Suggestion


class ExtractorBase:
    def __init__(self, extraction_identifier: ExtractionIdentifier):
        self.extraction_identifier = extraction_identifier

    @abstractmethod
    def create_model(self, extraction_data: ExtractionData) -> tuple[bool, str]:
        pass

    @abstractmethod
    def get_suggestions(self, predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        pass

    @abstractmethod
    def is_used(self) -> bool:
        pass
