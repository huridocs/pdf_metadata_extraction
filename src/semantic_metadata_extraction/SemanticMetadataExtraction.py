import os
from os.path import exists, join
from typing import List, Type

from config import config_logger, DATA_PATH
from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.Method import Method
from semantic_metadata_extraction.methods.DateParserMethod import DateParserMethod
from semantic_metadata_extraction.methods.MT5TrueCaseEnglishSpanishMethod import MT5TrueCaseEnglishSpanishMethod
from semantic_metadata_extraction.methods.RegexMethod import RegexMethod
from semantic_metadata_extraction.methods.SameInputOutputMethod import SameInputOutputMethod


class SemanticMetadataExtraction:
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

    METHODS: List[Type[Method]] = [SameInputOutputMethod, RegexMethod, DateParserMethod, MT5TrueCaseEnglishSpanishMethod]

    def __init__(self, tenant: str, property_name: str):
        self.tenant = tenant
        self.property_name = property_name

    def create_model(self, semantic_extraction_data: List[SemanticExtractionData]):
        if len(semantic_extraction_data) < 2:
            config_logger.info("Best method SameInputOutputMethod because no samples")
            return

        best_method_instance = self.get_best_method(semantic_extraction_data)
        best_method_instance.train(semantic_extraction_data)

    def get_best_method(self, semantic_extraction_data: List[SemanticExtractionData]):
        performance_semantic_extraction_data = [x for x in semantic_extraction_data if x.text]
        best_performance = 0
        best_method_instance = self.METHODS[0](self.tenant, self.property_name)
        for method in self.METHODS[:-1]:
            method_instance = method(self.tenant, self.property_name)
            performance, _ = method_instance.performance(performance_semantic_extraction_data, 30)
            if performance == 100:
                config_logger.info(f"Best method {method_instance.get_name()} with {performance}%")
                return method_instance

            if performance > best_performance:
                best_performance = performance
                best_method_instance = method_instance

        return self.decide_best_method_or_t5(best_performance, best_method_instance, performance_semantic_extraction_data)

    def decide_best_method_or_t5(
        self,
        best_performance: float,
        best_method_instance: Method,
        semantic_extraction_data: List[SemanticExtractionData],
    ):
        if best_performance > 85:
            config_logger.info(f"Best method {best_method_instance.get_name()} with {best_performance}%")
            return best_method_instance

        t5 = MT5TrueCaseEnglishSpanishMethod(self.tenant, self.property_name)

        if best_performance < 60:
            config_logger.info(f"Best method {t5.get_name()} because the others were bad")
            return t5

        performance, _ = t5.performance(semantic_extraction_data, 30)

        if performance > best_performance:
            config_logger.info(f"Best method {t5.get_name()} with {performance}%")
            return t5

        config_logger.info(f"Best method {best_method_instance.get_name()} with {best_performance}%")
        return best_method_instance

    def get_semantic_predictions(self, segments_text: List[str]) -> List[str]:
        for method in self.METHODS:
            method_instance = method(self.tenant, self.property_name)
            method_path = join(DATA_PATH, self.tenant, self.property_name, method_instance.get_name())
            config_logger.info(f"Checking {method_path}")

            if exists(method_path):
                config_logger.info(f"Predicting {len(segments_text)} documents with {method_instance.get_name()}")
                return method_instance.predict(segments_text)

        config_logger.info(f"Predicting {len(segments_text)} documents with SameInputOutputMethod")
        return self.METHODS[0](self.tenant, self.property_name).predict(segments_text)

    def remove_models(self):
        for method in self.METHODS:
            method_instance = method(self.tenant, self.property_name)
            method_instance.remove_model()
