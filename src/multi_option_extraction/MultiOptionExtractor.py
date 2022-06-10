import json
from os import makedirs
from os.path import join, exists
from pathlib import Path
from typing import List

import fuzzywuzzy.fuzz

from ServiceConfig import ServiceConfig
from data.Option import Option
from multi_option_extraction.MultiOptionExtractionData import MultiOptionExtractionData, MultiOptionExtractionSample
from semantic_metadata_extraction.SemanticMetadataExtraction import SemanticMetadataExtraction


class MultiOptionExtractor:
    def __init__(self, tenant: str, property_name: str):
        self.tenant = tenant
        self.property_name = property_name
        self.service_config = ServiceConfig()

        self.base_path = join(self.service_config.docker_volume_path, tenant, property_name, "multi_option_extractor")
        self.options_path = join(self.base_path, "options.json")
        self.multi_value_path = join(self.base_path, "multi_value.json")
        self.semantic_metadata_extraction = SemanticMetadataExtraction(self.tenant, self.property_name)

        self.options = list()
        self.multi_value = False

    def create_model(self, multi_option_extraction_data: MultiOptionExtractionData):
        self.save_json(self.options_path, [x.dict() for x in multi_option_extraction_data.options])
        self.save_json(self.multi_value_path, multi_option_extraction_data.multi_value)
        self.semantic_metadata_extraction.create_model(multi_option_extraction_data.to_semantic_extraction_data())
        return True, ""

    @staticmethod
    def save_json(path: str, data: any):
        if not exists(Path(path).parent):
            makedirs(Path(path).parent)

        with open(path, "w") as file:
            json.dump(data, file)

    def get_multi_option_predictions(self, texts: List[str]) -> List[MultiOptionExtractionSample]:
        self.load_options()
        if self.semantic_metadata_extraction.exists_model():
            semantic_predictions = self.semantic_metadata_extraction.get_semantic_predictions(texts)
            return [self.get_options_semantics(text, prediction) for text, prediction in zip(texts, semantic_predictions)]

        return [self.get_options_fuzzy_wuzzy(text) for text in texts]

    def load_options(self):
        with open(self.options_path, "r") as file:
            self.options = [Option(**x) for x in json.load(file)]
        with open(self.multi_value_path, "r") as file:
            self.multi_value = json.load(file)

    def get_options_fuzzy_wuzzy(self, text):
        ratios = [fuzzywuzzy.fuzz.partial_ratio(text.lower(), x.label.lower()) for x in self.options]

        if max(ratios) < 95:
            return MultiOptionExtractionSample(text=text, options=[])

        if not self.multi_value:
            best_ratio_index = ratios.index(max(ratios))
            return MultiOptionExtractionSample(text=text, options=[self.options[best_ratio_index]])

        options: List[Option] = list()
        for index, ratio in enumerate(ratios):
            if ratio > 95:
                options.append(self.options[index])

        return MultiOptionExtractionSample(text=text, options=options)

    def get_options_semantics(self, text, prediction_text: str):
        options = list()
        options_labels = [x.label for x in self.options]

        for predicted_option in prediction_text.split(' ; '):
            if predicted_option in options_labels:
                options.append(self.options[options_labels.index(predicted_option)])
                continue

        if prediction_text and not options:
            options = self.get_options_fuzzy_wuzzy(prediction_text)

        if not self.multi_value and options:
            return MultiOptionExtractionSample(text=text, options=[options[0]])

        return MultiOptionExtractionSample(text=text, options=options)

