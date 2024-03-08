import json
from os import makedirs
from os.path import join, exists
from pathlib import Path
from typing import Type

from config import DATA_PATH, config_logger
from data.Option import Option
from data.SemanticPredictionData import SemanticPredictionData
from multi_option_extraction.MultiOptionExtractionData import MultiOptionExtractionData, MultiOptionExtractionSample
from multi_option_extraction.TextToMultiOptionMethod import TextToMultiOptionMethod
from multi_option_extraction.text_to_multi_option_methods.FastTextMethod import FastTextMethod
from multi_option_extraction.text_to_multi_option_methods.TfIdfMethod import TfIdfMethod


class MultiOptionExtractor:
    METHODS: list[Type[TextToMultiOptionMethod]] = [FastTextMethod, TfIdfMethod]

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

        self.save_json(self.options_path, [x.model_dump() for x in multi_option_extraction_data.options])
        self.save_json(self.multi_value_path, multi_option_extraction_data.multi_value)

        method = self.get_best_method(multi_option_extraction_data)

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
        method = self.get_predictions_method()
        options_predictions = method.predict(semantic_predictions_data)
        for semantic_prediction_data, prediction in zip(semantic_predictions_data, options_predictions):
            multi_option_extraction_sample = MultiOptionExtractionSample(
                pdf_tags=semantic_prediction_data.pdf_tags, values=prediction
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

    def get_best_method(self, multi_option_extraction_data: MultiOptionExtractionData):
        best_method_instance = self.METHODS[0](self.tenant, self.extraction_id, self.options, self.multi_value)

        if len(self.METHODS) == 1:
            return best_method_instance

        samples = [sample for sample in multi_option_extraction_data.samples if sample.texts]
        performance_multi_option_extraction_data = MultiOptionExtractionData(
            multi_value=self.multi_value, options=self.options, samples=samples
        )

        best_performance = 0
        for method in self.METHODS:
            method_instance = method(self.tenant, self.extraction_id, self.options, self.multi_value)
            config_logger.info(f"\nChecking {method_instance.get_name()}")
            performance = method_instance.performance(performance_multi_option_extraction_data, 30)
            config_logger.info(f"\nPerformance {method_instance.get_name()}: {performance}%")
            if performance == 100:
                config_logger.info(f"\nBest method {method_instance.get_name()} with {performance}%")
                return method_instance

            if performance > best_performance:
                best_performance = performance
                best_method_instance = method_instance

        return best_method_instance

    @staticmethod
    def exist_model(tenant, extraction_id):
        multi_option_extractor = MultiOptionExtractor(tenant, extraction_id)
        multi_option_extractor.load_options()
        return len(multi_option_extractor.options) > 0

    def get_predictions_method(self):
        for method in self.METHODS:
            method_instance = method(self.tenant, self.extraction_id, self.options, self.multi_value)
            method_path = join(DATA_PATH, self.tenant, self.extraction_id, method_instance.get_name())
            config_logger.info(f"Checking {method_path}")

            if exists(method_path):
                config_logger.info(f"Predicting with {method_instance.get_name()}")

                return method_instance

        default_method = self.METHODS[0](self.tenant, self.extraction_id, self.options, self.multi_value)
        config_logger.info(f"Predicting with {default_method.get_name()}")
        return default_method
