import json
import os
import shutil
from abc import ABC, abstractmethod
from os import makedirs
from os.path import exists, join
from pathlib import Path


from config import config_logger, DATA_PATH
from data.PdfTagData import PdfTagData
from data.SemanticExtractionData import SemanticExtractionData
from data.SemanticPredictionData import SemanticPredictionData


class Method(ABC):
    def __init__(self, tenant: str, extraction_id: str):
        self.tenant = tenant
        self.extraction_id = extraction_id
        self.base_path = join(DATA_PATH, tenant, extraction_id)

        if not exists(self.base_path):
            os.makedirs(self.base_path)

    @abstractmethod
    def performance(self, semantic_extraction_data: list[SemanticExtractionData], training_set_length: int):
        pass

    @abstractmethod
    def train(self, semantic_extraction_data: list[SemanticExtractionData]):
        pass

    @abstractmethod
    def predict(self, pdf_tags: list[SemanticPredictionData]) -> list[str]:
        pass

    def get_name(self):
        return self.__class__.__name__

    def save_json(self, file_name: str, data: any):
        path = join(self.base_path, self.get_name(), file_name)
        if not exists(Path(path).parent):
            makedirs(Path(path).parent)

        with open(path, "w") as file:
            json.dump(data, file)

    def log_performance_sample(self, semantic_extractions_data: list[SemanticExtractionData], predictions: list[str]):
        config_logger.info(f"Performance predictions for {self.get_name()}")
        for i, semantic_extraction_data, prediction in zip(range(len(predictions)), semantic_extractions_data, predictions):
            if i >= 5:
                break

            config_logger.info("prediction: " + prediction)
            config_logger.info("truth     : " + semantic_extraction_data.text)
            config_logger.info("text      : " + self.get_text_from_pdf_tags(semantic_extraction_data.pdf_data))

    def load_json(self, file_name: str):
        path = join(self.base_path, self.get_name(), file_name)

        with open(path, "r") as file:
            return json.load(file)

    def remove_model(self):
        shutil.rmtree(join(self.base_path, self.get_name()), ignore_errors=True)

    @staticmethod
    def get_train_test(
        semantic_extraction_data: list[SemanticExtractionData], training_set_length: int
    ) -> (list[SemanticExtractionData], list[SemanticExtractionData]):
        if len(semantic_extraction_data) >= 2 * training_set_length:
            train = semantic_extraction_data[:training_set_length]
            test = semantic_extraction_data[training_set_length:]
            return train, test

        if len(semantic_extraction_data) <= 10:
            return semantic_extraction_data, semantic_extraction_data

        train_amount = len(semantic_extraction_data) // 2
        training_set = semantic_extraction_data[:train_amount]
        training_set = training_set[:training_set_length]

        testing_set = semantic_extraction_data[train_amount:]
        return training_set, testing_set

    @staticmethod
    def get_segments_texts_without_breaks(semantic_data: list[SemanticExtractionData]) -> list[str]:
        return [Method.get_text_from_pdf_tags(x.pdf_tags) for x in semantic_data]

    @staticmethod
    def get_text_from_pdf_tags(pdf_tags_data: list[PdfTagData]) -> str:
        return " ".join([pdf_tag_data.text.strip() for pdf_tag_data in pdf_tags_data])

    @staticmethod
    def clean(text):
        return " ".join(text.replace("\n", " ").strip().split())
