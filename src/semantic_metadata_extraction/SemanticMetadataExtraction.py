import os
from os.path import exists, join
from typing import List, Type

from ServiceConfig import ServiceConfig
from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.Method import Method
from semantic_metadata_extraction.methods.DateParserMethod import DateParserMethod
from semantic_metadata_extraction.methods.MT5TrueCaseEnglishSpanishMethod import MT5TrueCaseEnglishSpanishMethod
from semantic_metadata_extraction.methods.RegexMethod import RegexMethod
from semantic_metadata_extraction.methods.SameInputOutputMethod import SameInputOutputMethod


class SemanticMetadataExtraction:
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

    METHODS: List[Type[Method]] = [
        SameInputOutputMethod,
        RegexMethod,
        DateParserMethod,
        MT5TrueCaseEnglishSpanishMethod,
    ]

    def __init__(self, tenant: str, property_name: str):
        self.tenant = tenant
        self.property_name = property_name
        service_config = ServiceConfig()
        self.docker_volume = service_config.docker_volume_path

    def create_model(self, semantic_extraction_data: List[SemanticExtractionData]):
        if len(semantic_extraction_data) < 2:
            return

        best_method_instance = self.get_best_method(semantic_extraction_data)
        best_method_instance.train(semantic_extraction_data)

    def get_best_method(self, semantic_extraction_data):
        best_performance = 0
        best_method_instance = self.METHODS[0](self.tenant, self.property_name)
        for method in self.METHODS:
            performance, _ = method(self.tenant, self.property_name).performance(semantic_extraction_data, 30)
            if performance == 100:
                return method(self.tenant, self.property_name)

            if performance > best_performance:
                best_performance = performance
                best_method_instance = method(self.tenant, self.property_name)

        return best_method_instance

    def get_semantic_predictions(self, segments_text: List[str]) -> List[str]:
        for method in self.METHODS:
            method_instance = method(self.tenant, self.property_name)
            if exists(join(self.docker_volume, self.tenant, self.property_name, method_instance.get_name())):
                return method_instance.predict(segments_text)

        return self.METHODS[0](self.tenant, self.property_name).predict(segments_text)

    def remove_models(self):
        for method in self.METHODS:
            method_instance = method(self.tenant, self.property_name)
            method_instance.remove_model()
