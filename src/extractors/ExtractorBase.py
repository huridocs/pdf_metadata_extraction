import json
import random
from abc import abstractmethod
from os import makedirs
from os.path import exists
from pathlib import Path

from data.ExtractionData import ExtractionData
from data.ExtractionIdentifier import ExtractionIdentifier
from data.PredictionSample import PredictionSample
from data.Suggestion import Suggestion
from data.TrainingSample import TrainingSample


class ExtractorBase:
    def __init__(self, extraction_identifier: ExtractionIdentifier):
        self.extraction_identifier = extraction_identifier

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def create_model(self, extraction_data: ExtractionData) -> tuple[bool, str]:
        pass

    @abstractmethod
    def get_suggestions(self, predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        pass

    @abstractmethod
    def is_valid(self, extraction_data: ExtractionData) -> bool:
        pass

    @staticmethod
    def get_train_test_sets(extraction_data: ExtractionData, seed: int = 22) -> (ExtractionData, ExtractionData):
        if len(extraction_data.samples) < 15:
            return extraction_data, extraction_data

        train_size = int(len(extraction_data.samples) * 0.8)
        random.seed(seed)

        train_set: list[TrainingSample] = random.sample(extraction_data.samples, k=train_size)[:80]
        test_set: list[TrainingSample] = [x for x in extraction_data.samples if x not in train_set][:30]

        train_data = ExtractionData(
            samples=train_set,
            options=extraction_data.options,
            multi_value=extraction_data.multi_value,
            extraction_identifier=extraction_data.extraction_identifier,
        )

        test_data = ExtractionData(
            samples=test_set,
            options=extraction_data.options,
            multi_value=extraction_data.multi_value,
            extraction_identifier=extraction_data.extraction_identifier,
        )

        return train_data, test_data

    @staticmethod
    def save_json(path: str, data: any):
        if not exists(Path(path).parent):
            makedirs(Path(path).parent)

        with open(path, "w") as file:
            json.dump(data, file)