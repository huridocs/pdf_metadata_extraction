import json
import os
import shutil
from abc import ABC, abstractmethod
from os import makedirs
from os.path import exists, join
from pathlib import Path

from config import DATA_PATH
from data.Option import Option
from data.PdfTagData import PdfTagData
from data.SemanticPredictionData import SemanticPredictionData
from multi_option_extraction.MultiOptionExtractionData import MultiOptionExtractionData


class MultiOptionMethod(ABC):
    def __init__(self, tenant: str, extraction_id: str, options: list[Option], multi_value: bool):
        self.multi_value = multi_value
        self.options = options
        self.tenant = tenant
        self.extraction_id = extraction_id
        self.base_path = join(DATA_PATH, tenant, extraction_id)

        if not exists(self.base_path):
            os.makedirs(self.base_path)


    @abstractmethod
    def train(self, multi_option_extraction_data: MultiOptionExtractionData):
        pass

    @abstractmethod
    def predict(self, semantic_predictions_data: list[SemanticPredictionData]) -> list[list[Option]]:
        pass

    def get_name(self):
        return self.__class__.__name__

    def save_json(self, file_name: str, data: any):
        path = join(self.base_path, self.get_name(), file_name)
        if not exists(Path(path).parent):
            makedirs(Path(path).parent)

        with open(path, "w") as file:
            json.dump(data, file)

    def load_json(self, file_name: str):
        path = join(self.base_path, self.get_name(), file_name)

        with open(path, "r") as file:
            return json.load(file)

    def remove_model(self):
        shutil.rmtree(join(self.base_path, self.get_name()), ignore_errors=True)

    @staticmethod
    def get_text_from_pdf_tags(pdf_tags_data: list[PdfTagData]) -> str:
        return " ".join([pdf_tag_data.text.strip() for pdf_tag_data in pdf_tags_data])
