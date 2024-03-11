import json
import math
import os
import shutil
from abc import ABC, abstractmethod
from os import makedirs
from os.path import exists, join
from pathlib import Path

import torch
from numpy import argmax

from config import DATA_PATH
from data.Option import Option
from data.SemanticPredictionData import SemanticPredictionData
from multi_option_extraction.data.MultiOptionData import MultiOptionData


class MultiLabelMethod(ABC):
    def __init__(self, tenant: str, extraction_id: str, options: list[Option], multi_value: bool):
        self.tenant = tenant
        self.extraction_id = extraction_id
        self.options = options
        self.multi_value = multi_value
        self.base_path = join(DATA_PATH, tenant, extraction_id)

        if not exists(self.base_path):
            os.makedirs(self.base_path)

    @abstractmethod
    def train(self, multi_option_data: MultiOptionData):
        pass

    @abstractmethod
    def predict(self, multi_option_data: MultiOptionData) -> list[list[Option]]:
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

    def get_texts_labels(self, multi_option_data: MultiOptionData) -> (list[str], list[list[int]]):
        texts = list()
        for sample in multi_option_data.samples:
            texts.append(" ".join([x.text_content.strip() for x in sample.pdf_data.pdf_data_segments]))

        labels = self.get_one_hot_encoding(multi_option_data)
        return texts, labels

    @staticmethod
    def get_train_test(
        multi_option_data: MultiOptionData, training_set_length: int
    ) -> (list[MultiOptionData], list[MultiOptionData]):
        if len(multi_option_data.samples) >= 2 * training_set_length:
            train_samples = multi_option_data.samples[:training_set_length]
            train = MultiOptionData(samples=train_samples, options=multi_option_data.options, multi_value=False)
            test_samples = multi_option_data.samples[training_set_length:]
            test = MultiOptionData(samples=test_samples, options=multi_option_data.options, multi_value=False)
            return train, test

        if len(multi_option_data.samples) <= 10:
            return multi_option_data, multi_option_data

        train_amount = len(multi_option_data.samples) // 2
        train_samples = multi_option_data.samples[:train_amount]
        training_set = MultiOptionData(samples=train_samples, options=[], multi_value=False)
        training_set.samples = training_set.samples[:training_set_length]
        test_samples = multi_option_data.samples[train_amount:]
        testing_set = MultiOptionData(samples=test_samples, options=[], multi_value=False)
        return training_set, testing_set

    def get_extraction_data_from_samples(self, samples: list[SemanticPredictionData]):
        return MultiOptionData(samples=samples, options=self.options, multi_value=False)

    def predictions_to_options_list(self, predictions) -> list[list[Option]]:
        return [self.one_prediction_to_option_list(prediction) for prediction in predictions]

    def one_prediction_to_option_list(self, prediction) -> list[Option]:
        if not self.multi_value:
            best_score_index = argmax(prediction)
            return [self.options[best_score_index]] if prediction[best_score_index] > 0.5 else []

        return [self.options[i] for i, value in enumerate(prediction) if value > 0.5]

    def get_one_hot_encoding(self, multi_option_data: MultiOptionData):
        options_ids = [option.id for option in self.options]
        one_hot_encoding = list()
        for sample in multi_option_data.samples:
            one_hot_encoding.append([0] * len(options_ids))
            for option in sample.values:
                one_hot_encoding[-1][options_ids.index(option.id)] = 1
        return one_hot_encoding

    @staticmethod
    def get_batch_size(multi_option_data: MultiOptionData):
        samples_count = len(multi_option_data.samples)
        batch_size_by_samples = math.ceil(samples_count / 100)
        memory_available = torch.cuda.get_device_properties(0).total_memory / 1000000000
        limit_batch = max(int(memory_available / 5), 1)
        return min(limit_batch, batch_size_by_samples)

    @staticmethod
    def get_max_steps(multi_option_data: MultiOptionData):
        steps = math.ceil(23 * len(multi_option_data.samples) / 200) * 200
        return min(steps, 2000)
