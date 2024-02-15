from abc import ABC, abstractmethod

from paragraph_extraction_trainer.Paragraph import Paragraph

from data.Option import Option
from data.SemanticPredictionData import SemanticPredictionData


class TextExtractionMethod(ABC):
    def __init__(self, pdf_paragraphs: list[Paragraph], options: list[Option], multi_value: bool):
        self.multi_value = multi_value
        self.options = options
        self.pdf_paragraphs = pdf_paragraphs

    @abstractmethod
    def get_text(self) -> list[SemanticPredictionData]:
        pass
