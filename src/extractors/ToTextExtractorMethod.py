import json
import os
import shutil
from abc import abstractmethod
from os.path import join, exists
from pathlib import Path

from config import config_logger
from data.ExtractionData import ExtractionData
from data.ExtractionIdentifier import ExtractionIdentifier
from data.PredictionSample import PredictionSample
from extractors.ExtractorBase import ExtractorBase


class ToTextExtractorMethod:

    def __init__(self, extraction_identifier: ExtractionIdentifier):
        self.extraction_identifier = extraction_identifier
        os.makedirs(self.extraction_identifier.get_path(), exist_ok=True)

    def get_name(self):
        return self.__class__.__name__

    def save_json(self, file_name: str, data: any):
        path = join(self.extraction_identifier.get_path(), self.get_name(), file_name)
        if not exists(Path(path).parent):
            os.makedirs(Path(path).parent)

        with open(path, "w") as file:
            json.dump(data, file)

    def load_json(self, file_name: str):
        path = join(self.extraction_identifier.get_path(), self.get_name(), file_name)

        with open(path, "r") as file:
            return json.load(file)

    def remove_model(self):
        shutil.rmtree(join(self.extraction_identifier.get_path(), self.get_name()), ignore_errors=True)

    @abstractmethod
    def train(self, extraction_data: ExtractionData):
        pass

    @abstractmethod
    def predict(self, predictions_samples: list[PredictionSample]) -> list[str]:
        pass

    @staticmethod
    def clean_text(text: str) -> str:
        return " ".join(text.split())

    def performance(self, extraction_data: ExtractionData) -> float:
        if not extraction_data.samples:
            return 0

        performance_train_set, performance_test_set = ExtractorBase.get_train_test_sets(extraction_data)

        self.train(performance_train_set)
        samples = performance_test_set.samples
        predictions = self.predict([PredictionSample(pdf_data=x.pdf_data, tags_texts=x.tags_texts) for x in samples])

        self.log_performance_sample(extraction_data=performance_test_set, predictions=predictions)

        correct = [
            sample
            for sample, prediction in zip(performance_test_set.samples, predictions)
            if self.clean_text(sample.labeled_data.label_text) == self.clean_text(prediction)
        ]
        self.remove_model()
        return 100 * len(correct) / len(performance_test_set.samples)

    def log_performance_sample(self, extraction_data: ExtractionData, predictions: list[str]):
        config_logger.info(f"Performance predictions for {self.get_name()}")
        for i, (training_sample, prediction) in enumerate(zip(extraction_data.samples, predictions)):
            if i >= 3:
                break

            config_logger.info("prediction: " + prediction)
            config_logger.info("truth     : " + training_sample.labeled_data.label_text)
            config_logger.info("text      : " + " ".join(training_sample.tags_texts))
