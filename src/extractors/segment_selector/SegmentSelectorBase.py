from abc import ABC, abstractmethod

from data.ExtractionIdentifier import ExtractionIdentifier
from data.PdfData import PdfData


class SegmentSelectorBase(ABC):

    def __init__(self, extraction_identifier: ExtractionIdentifier, method_name: str = ""):
        self.extraction_identifier = extraction_identifier
        self.method_name = method_name

    @abstractmethod
    def prepare_model_folder(self):
        pass

    @abstractmethod
    def get_predictions_for_performance(self, training_set: list[PdfData], test_set: list[PdfData]) -> list[int]:
        pass
