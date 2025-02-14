from typing import Optional

import pymongo
from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from pydantic import BaseModel
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.PredictionData import PredictionData
from trainable_entity_extractor.data.Suggestion import Suggestion

from config import MONGO_HOST, MONGO_PORT
from domain.ParagraphExtractionData import ParagraphExtractionData
from ports.PersistenceRepository import PersistenceRepository


class MongoPersistenceRepository(PersistenceRepository):

    def __init__(self):
        self.mongodb_client = pymongo.MongoClient(f"{MONGO_HOST}:{MONGO_PORT}")
        self.mongo_db = self.mongodb_client["pdf_metadata_extraction"]

    def close(self):
        self.mongodb_client.close()

    @staticmethod
    def get_filter(extraction_identifier: ExtractionIdentifier):
        return {
            "run_name": extraction_identifier.run_name,
            "extraction_name": extraction_identifier.extraction_name,
        }

    @staticmethod
    def inject_extractor_identifier(extraction_identifier: ExtractionIdentifier, data: dict):
        data["run_name"] = extraction_identifier.run_name
        data["extraction_name"] = extraction_identifier.extraction_name
        return data

    def save_data(self, extraction_identifier: ExtractionIdentifier, data: BaseModel, collection_name: str):
        data_dict = data.model_dump()
        data_dict = self.inject_extractor_identifier(extraction_identifier, data_dict)
        self.mongo_db[collection_name].insert_one(data_dict)

    def save_prediction_data(self, extraction_identifier: ExtractionIdentifier, prediction_data: PredictionData):
        self.save_data(extraction_identifier, prediction_data, "prediction_data")

    def load_prediction_data(self, extraction_identifier: ExtractionIdentifier) -> list[PredictionData]:
        data = self.mongo_db.prediction_data.find(self.get_filter(extraction_identifier))
        prediction_data = [PredictionData(**document) for document in data]
        self.mongo_db.prediction_data.delete_many(self.get_filter(extraction_identifier))
        return prediction_data

    def save_labeled_data(self, extraction_identifier: ExtractionIdentifier, labeled_data: LabeledData):
        self.save_data(extraction_identifier, labeled_data, "labeled_data")

    def delete_labeled_data(self, extraction_identifier: ExtractionIdentifier):
        self.mongo_db.labeled_data.delete_many(self.get_filter(extraction_identifier))

    def load_labeled_data(self, extraction_identifier: ExtractionIdentifier) -> list[LabeledData]:
        data = self.mongo_db.labeled_data.find(self.get_filter(extraction_identifier))
        return [LabeledData(**document) for document in data]

    def save_suggestions(self, extraction_identifier: ExtractionIdentifier, suggestions: list[Suggestion]):
        for suggestion in suggestions:
            self.save_data(extraction_identifier, suggestion, "suggestions")

    def load_suggestions(self, extraction_identifier: ExtractionIdentifier) -> list[Suggestion]:
        suggestions: list[Suggestion] = list()

        for document in self.mongo_db.suggestions.find(self.get_filter(extraction_identifier)):
            suggestions.append(Suggestion(**document))

        self.mongo_db.suggestions.delete_many(self.get_filter(extraction_identifier))

        return suggestions

    def save_paragraph_extraction_data(
        self, extraction_identifier: ExtractionIdentifier, paragraph_extraction_data: ParagraphExtractionData
    ):
        self.save_data(extraction_identifier, paragraph_extraction_data, "paragraph_extraction_data")

    def load_paragraph_extraction_data(
        self, extraction_identifier: ExtractionIdentifier
    ) -> Optional[ParagraphExtractionData]:
        data = self.mongo_db.paragraph_extraction_data.find_one(self.get_filter(extraction_identifier))
        if data is None:
            return None
        return ParagraphExtractionData(**data)

    def save_paragraphs_from_language(
        self, extraction_identifier: ExtractionIdentifier, paragraphs_from_languages: ParagraphsFromLanguage
    ):
        self.save_data(extraction_identifier, paragraphs_from_languages, "paragraphs_from_languages")

    def load_paragraphs_from_languages(self, extraction_identifier: ExtractionIdentifier) -> list[ParagraphsFromLanguage]:
        data = self.mongo_db.paragraphs_from_languages.find(self.get_filter(extraction_identifier))
        paragraphs = [ParagraphsFromLanguage(**document) for document in data]
        self.mongo_db.paragraphs_from_languages.delete_many(self.get_filter(extraction_identifier))
        return paragraphs

    def delete_paragraphs_from_languages(self, extraction_identifier: ExtractionIdentifier):
        self.mongo_db.paragraphs_from_languages.delete_many(self.get_filter(extraction_identifier))

    def delete_prediction_data(self, extraction_identifier: ExtractionIdentifier, filters: list[dict[str, str]]):
        for one_filter in filters:
            self.mongo_db.suggestions.delete_many({**self.get_filter(extraction_identifier), **one_filter})
