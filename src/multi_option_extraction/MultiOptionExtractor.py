import json
from os import makedirs
from os.path import join, exists
from pathlib import Path

import fuzzywuzzy.fuzz

from config import DATA_PATH
from data.Option import Option
from data.SemanticPredictionData import SemanticPredictionData
from multi_option_extraction.MultiOptionExtractionData import MultiOptionExtractionData, MultiOptionExtractionSample
from multi_option_extraction.methods.FastTextMethod import FastTextMethod


class MultiOptionExtractor:
    def __init__(self, tenant: str, extraction_id: str):
        self.tenant = tenant
        self.extraction_id = extraction_id

        self.base_path = join(DATA_PATH, tenant, extraction_id, "multi_option_extractor")
        self.options_path = join(self.base_path, "options.json")
        self.multi_value_path = join(self.base_path, "multi_value.json")

        self.options: list[Option] = list()
        self.multi_value = False

    def create_model(self, multi_option_extraction_data: MultiOptionExtractionData):
        self.options = multi_option_extraction_data.options
        self.multi_value = multi_option_extraction_data.multi_value

        self.save_json(self.options_path, [x.dict() for x in multi_option_extraction_data.options])
        self.save_json(self.multi_value_path, multi_option_extraction_data.multi_value)

        method = FastTextMethod(self.tenant, self.extraction_id, self.options, self.multi_value)

        method.train(multi_option_extraction_data)

        return True, ""

    @staticmethod
    def save_json(path: str, data: any):
        if not exists(Path(path).parent):
            makedirs(Path(path).parent)

        with open(path, "w") as file:
            json.dump(data, file)

    def get_multi_option_predictions(
        self, semantic_predictions_data: list[SemanticPredictionData]
    ) -> list[MultiOptionExtractionSample]:
        self.load_options()

        multi_option_extraction_samples = list()
        method = FastTextMethod(self.tenant, self.extraction_id, self.options, self.multi_value)
        options_predictions = method.predict(semantic_predictions_data)
        for semantic_prediction_data, prediction in zip(semantic_predictions_data, options_predictions):
            multi_option_extraction_sample = MultiOptionExtractionSample(
                pdf_tags=semantic_prediction_data.pdf_tags, options=prediction
            )
            multi_option_extraction_samples.append(multi_option_extraction_sample)

        return multi_option_extraction_samples

    def load_options(self):
        if not exists(self.options_path) or not exists(self.multi_value_path):
            return

        with open(self.options_path, "r") as file:
            self.options = [Option(**x) for x in json.load(file)]

        with open(self.multi_value_path, "r") as file:
            self.multi_value = json.load(file)

    @staticmethod
    def exist_model(tenant, extraction_id):
        multi_option_extractor = MultiOptionExtractor(tenant, extraction_id)
        multi_option_extractor.load_options()
        return len(multi_option_extractor.options) > 0
