import json
import os
import shutil
from abc import ABC, abstractmethod
from os import makedirs
from os.path import exists, join
from pathlib import Path
from typing import List

from config import config_logger, DATA_PATH
from data.SemanticExtractionData import SemanticExtractionData


class Method(ABC):
    def __init__(self, tenant: str, property_name: str):
        self.tenant = tenant
        self.property_name = property_name
        self.base_path = join(DATA_PATH, tenant, property_name)

        if not exists(self.base_path):
            os.makedirs(self.base_path)

    @abstractmethod
    def performance(self, semantic_extraction_data: List[SemanticExtractionData], training_set_length: int):
        pass

    @abstractmethod
    def train(self, semantic_extraction_data: List[SemanticExtractionData]):
        pass

    @abstractmethod
    def predict(self, texts: List[str]) -> List[str]:
        pass

    def get_name(self):
        return self.__class__.__name__

    def save_json(self, file_name: str, data: any):
        path = join(self.base_path, self.get_name(), file_name)
        if not exists(Path(path).parent):
            makedirs(Path(path).parent)

        with open(path, "w") as file:
            json.dump(data, file)

    def log_performance_sample(self, semantic_extractions_data: List[SemanticExtractionData], predictions: List[str]):
        config_logger.info(f"Performance predictions for {self.get_name()}")
        for i, semantic_extraction_data, prediction in zip(range(len(predictions)), semantic_extractions_data, predictions):
            if i >= 5:
                break

            config_logger.info("prediction: " + prediction)
            config_logger.info("truth     : " + semantic_extraction_data.text)
            config_logger.info("text      : " + semantic_extraction_data.segment_text)

    def load_json(self, file_name: str):
        path = join(self.base_path, self.get_name(), file_name)

        with open(path, "r") as file:
            return json.load(file)

    def remove_model(self):
        shutil.rmtree(join(self.base_path, self.get_name()), ignore_errors=True)

    @staticmethod
    def get_train_test(
        semantic_extraction_data: List[SemanticExtractionData], training_set_length: int
    ) -> (List[SemanticExtractionData], List[SemanticExtractionData]):
        if len(semantic_extraction_data) >= 2 * training_set_length:
            train = Method.clean_semantic_extraction_data(semantic_extraction_data[:training_set_length])
            test = Method.clean_semantic_extraction_data(semantic_extraction_data[training_set_length:])
            return train, test

        if len(semantic_extraction_data) <= 10:
            cleaned_semantic_data = Method.clean_semantic_extraction_data(semantic_extraction_data)
            return cleaned_semantic_data, cleaned_semantic_data

        train_amount = len(semantic_extraction_data) // 2
        training_set = semantic_extraction_data[:train_amount]
        training_set = training_set[:training_set_length]
        training_set = Method.clean_semantic_extraction_data(training_set)

        testing_set = semantic_extraction_data[train_amount:]
        testing_set = Method.clean_semantic_extraction_data(testing_set)
        return training_set, testing_set

    @staticmethod
    def clean(text):
        return " ".join(text.replace("\n", " ").strip().split())

    @staticmethod
    def clean_semantic_extraction_data(semantic_extraction_data: List[SemanticExtractionData]):
        for semantic_extraction in semantic_extraction_data:
            semantic_extraction.segment_text = Method.clean(semantic_extraction.segment_text)
            semantic_extraction.text = Method.clean(semantic_extraction.text)

        return semantic_extraction_data
