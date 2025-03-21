from abc import abstractmethod, ABC

from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionData import PredictionData
from trainable_entity_extractor.domain.Suggestion import Suggestion

from domain.ParagraphExtractionData import ParagraphExtractionData


class PersistenceRepository(ABC):

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def save_prediction_data(self, extraction_identifier: ExtractionIdentifier, prediction_data: PredictionData):
        pass

    @abstractmethod
    def load_prediction_data(self, extraction_identifier: ExtractionIdentifier) -> list[PredictionData]:
        pass

    @abstractmethod
    def load_and_delete_prediction_data(
        self, extraction_identifier: ExtractionIdentifier, batch_size: int
    ) -> list[PredictionData]:
        pass

    @abstractmethod
    def save_labeled_data(self, extraction_identifier: ExtractionIdentifier, labeled_data: LabeledData):
        pass

    @abstractmethod
    def delete_labeled_data(self, extraction_identifier: ExtractionIdentifier):
        pass

    @abstractmethod
    def load_labeled_data(self, extraction_identifier: ExtractionIdentifier) -> list[LabeledData]:
        pass

    @abstractmethod
    def load_and_delete_labeled_data(
        self, extraction_identifier: ExtractionIdentifier, batch_size: int
    ) -> list[LabeledData]:
        pass

    @abstractmethod
    def save_suggestions(self, extraction_identifier: ExtractionIdentifier, suggestions: list[Suggestion]):
        pass

    @abstractmethod
    def load_suggestions(self, extraction_identifier: ExtractionIdentifier) -> list[Suggestion]:
        pass

    @abstractmethod
    def save_paragraph_extraction_data(
        self, extraction_identifier: ExtractionIdentifier, paragraph_extraction_data: ParagraphExtractionData
    ):
        pass

    @abstractmethod
    def load_paragraph_extraction_data(self, extraction_identifier: ExtractionIdentifier) -> ParagraphExtractionData:
        pass

    @abstractmethod
    def save_paragraphs_from_language(
        self, extraction_identifier: ExtractionIdentifier, paragraphs_from_languages: ParagraphsFromLanguage
    ):
        pass

    @abstractmethod
    def load_paragraphs_from_languages(self, extraction_identifier: ExtractionIdentifier) -> list[ParagraphsFromLanguage]:
        pass
