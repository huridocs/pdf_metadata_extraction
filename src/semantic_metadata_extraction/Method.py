import json
import os
from abc import ABC, abstractmethod
from os import makedirs
from os.path import exists, join
from pathlib import Path
from typing import List

from ServiceConfig import ServiceConfig
from data.SemanticExtractionData import SemanticExtractionData


class Method(ABC):
    def __init__(self, tenant: str, property_name: str):
        self.tenant = tenant
        self.property_name = property_name
        self.service_config = ServiceConfig()
        self.base_path = join(self.service_config.docker_volume_path, tenant, property_name)

        if not exists(self.base_path):
            os.makedirs(self.base_path)

    @abstractmethod
    def performance(self, semantic_extraction_data: List[SemanticExtractionData], training_set_length: int):
        pass

    @abstractmethod
    def train(self, semantic_extraction_data: List[SemanticExtractionData]):
        pass

    @abstractmethod
    def predict(self, semantic_extraction_data: List[str]) -> List[str]:
        pass

    def get_name(self):
        return self.__class__.__name__

    def save_json(self, file_name: str, data: any):
        path = join(self.base_path, file_name)
        if not exists(Path(path).parent):
            makedirs(Path(path).parent)

        with open(path, "w") as file:
            json.dump(data, file)

    def load_json(self, file_name: str):
        path = join(self.base_path, file_name)

        with open(path, "r") as file:
            return json.load(file)

    @staticmethod
    def get_train_test(
        semantic_extraction_data: List[SemanticExtractionData], training_set_length: int
    ) -> (List[SemanticExtractionData], List[SemanticExtractionData]):
        if len(semantic_extraction_data) < 2 * training_set_length:
            train_amount = len(semantic_extraction_data) // 2
            training_set = semantic_extraction_data[:train_amount]
            training_set = training_set[:training_set_length]
            testing_set = semantic_extraction_data[train_amount:]
            return training_set, testing_set

        return semantic_extraction_data[:training_set_length], semantic_extraction_data[training_set_length:]
