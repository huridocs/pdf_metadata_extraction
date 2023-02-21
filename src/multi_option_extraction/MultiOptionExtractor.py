import json
from os import makedirs
from os.path import join, exists
from pathlib import Path

import fuzzywuzzy.fuzz

from config import DATA_PATH
from data.Option import Option
from data.SemanticPredictionData import SemanticPredictionData
from multi_option_extraction.MultiOptionExtractionData import MultiOptionExtractionData, MultiOptionExtractionSample
from semantic_metadata_extraction.SemanticMetadataExtraction import SemanticMetadataExtraction


class MultiOptionExtractor:
    def __init__(self, tenant: str, property_name: str):
        self.tenant = tenant
        self.property_name = property_name

        self.base_path = join(DATA_PATH, tenant, property_name, "multi_option_extractor")
        self.options_path = join(self.base_path, "options.json")
        self.multi_value_path = join(self.base_path, "multi_value.json")
        self.semantic_metadata_extraction = SemanticMetadataExtraction(self.tenant, self.property_name)

        self.options: list[Option] = list()
        self.multi_value = False

    def create_model(self, multi_option_extraction_data: MultiOptionExtractionData):
        self.save_json(self.options_path, [x.dict() for x in multi_option_extraction_data.options])
        self.save_json(self.multi_value_path, multi_option_extraction_data.multi_value)
        self.semantic_metadata_extraction.remove_models()
        self.semantic_metadata_extraction.create_model(multi_option_extraction_data.to_semantic_extraction_data())
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
        semantic_predictions = self.semantic_metadata_extraction.get_semantic_predictions(semantic_predictions_data)

        multi_option_extraction_samples = list()
        for semantic_prediction_data, prediction in zip(semantic_predictions_data, semantic_predictions):
            multi_option_extraction_samples.append(self.get_options_semantics(semantic_prediction_data, prediction))

        return multi_option_extraction_samples

    def load_options(self):
        if not exists(self.options_path) or not exists(self.multi_value_path):
            return

        with open(self.options_path, "r") as file:
            self.options = [Option(**x) for x in json.load(file)]

        with open(self.multi_value_path, "r") as file:
            self.multi_value = json.load(file)

    def get_options_fuzzy(self, predicted_text: str):
        ratios = [fuzzywuzzy.fuzz.partial_ratio(predicted_text.lower(), x.label.lower()) for x in self.options]

        if max(ratios) < 95:
            return []

        if not self.multi_value:
            best_ratio_index = ratios.index(max(ratios))
            return [self.options[best_ratio_index]]

        options: list[Option] = list()
        for index, ratio in enumerate(ratios):
            if ratio > 95:
                options.append(self.options[index])

        return options

    def get_options_semantics(self, semantic_prediction_data: SemanticPredictionData, prediction_text: str):
        options = list()
        options_labels = [x.label for x in self.options]

        for predicted_option in prediction_text.split(" ; "):
            if predicted_option in options_labels:
                options.append(self.options[options_labels.index(predicted_option)])
                continue

        if prediction_text and not options:
            options = self.get_options_fuzzy(prediction_text)

        if not self.multi_value and options:
            return MultiOptionExtractionSample(pdf_tags=semantic_prediction_data.pdf_tags, options=[options[0]])

        return MultiOptionExtractionSample(pdf_tags=semantic_prediction_data.pdf_tags, options=options)

    @staticmethod
    def exist_model(tenant, property_name):
        multi_option_extractor = MultiOptionExtractor(tenant, property_name)
        multi_option_extractor.load_options()
        return len(multi_option_extractor.options) > 0
