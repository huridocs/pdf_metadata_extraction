import json
import os
import shutil
from abc import ABC, abstractmethod
from os import makedirs
from os.path import exists, join
from pathlib import Path

from sklearn.metrics import f1_score

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

    def performance(self, multi_option_extraction_data: MultiOptionExtractionData, training_set_length: int):
        if not multi_option_extraction_data.samples:
            return 0

        performance_train_set, performance_test_set = self.get_train_test(multi_option_extraction_data, training_set_length)

        self.train(performance_train_set)
        prediction_options = self.predict(performance_test_set.to_semantic_prediction_data())

        self.remove_model()
        return self.performance_f1_score(performance_test_set, prediction_options)

    def get_one_hot_encoding(self, multi_option_extraction_data: MultiOptionExtractionData):
        options_ids = [option.id for option in self.options]
        one_hot_encoding = list()
        for sample in multi_option_extraction_data.samples:
            one_hot_encoding.append([0] * len(options_ids))
            for option in sample.values:
                one_hot_encoding[-1][options_ids.index(option.id)] = 1
        return one_hot_encoding

    def performance_f1_score(self, multi_option_extraction_data: MultiOptionExtractionData, predictions: list[list[Option]]):
        truth_one_hot = self.get_one_hot_encoding(multi_option_extraction_data)

        predictions_one_hot = list()
        options_ids = [option.id for option in self.options]
        for prediction in predictions:
            predictions_one_hot.append([0] * len(options_ids))
            for option in prediction:
                predictions_one_hot[-1][options_ids.index(option.id)] = 1

        return 100 * f1_score(truth_one_hot, predictions_one_hot, average="macro")

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

    @staticmethod
    def get_train_test(
        multi_option_extraction_data: MultiOptionExtractionData, training_set_length: int
    ) -> (list[MultiOptionExtractionData], list[MultiOptionExtractionData]):
        if len(multi_option_extraction_data.samples) >= 2 * training_set_length:
            train = MultiOptionExtractionData(False, [], samples=multi_option_extraction_data.samples[:training_set_length])
            test = MultiOptionExtractionData(False, [], samples=multi_option_extraction_data.samples[training_set_length:])
            return train, test

        if len(multi_option_extraction_data.samples) <= 10:
            return multi_option_extraction_data, multi_option_extraction_data

        train_amount = len(multi_option_extraction_data.samples) // 2
        training_set = MultiOptionExtractionData(False, [], samples=multi_option_extraction_data.samples[:train_amount])
        training_set.samples = training_set.samples[:training_set_length]

        testing_set = MultiOptionExtractionData(False, [], samples=multi_option_extraction_data.samples[train_amount:])
        return training_set, testing_set

    def one_hot_to_options_list(self, predictions):
        prediction_options: list[list[Option]] = list()
        for prediction in predictions:
            prediction_options.append(list())
            for i, value in enumerate(prediction):
                if value > 0.5:
                    prediction_options[-1].append(self.options[i])

        return prediction_options
