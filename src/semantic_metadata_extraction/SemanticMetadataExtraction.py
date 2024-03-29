import os
from os.path import exists, join
from typing import Type

from config import config_logger
from data.ExtractionIdentifier import ExtractionIdentifier
from data.SemanticExtractionData import SemanticExtractionData
from data.SemanticPredictionData import SemanticPredictionData
from semantic_metadata_extraction.SemanticMethod import SemanticMethod
from semantic_metadata_extraction.methods.DateParserMethod import DateParserMethod
from semantic_metadata_extraction.methods.DateParserWithBreaksMethod import DateParserWithBreaksMethod
from semantic_metadata_extraction.methods.MT5TrueCaseEnglishSpanishMethod import MT5TrueCaseEnglishSpanishMethod
from semantic_metadata_extraction.methods.RegexMethod import RegexMethod
from semantic_metadata_extraction.methods.RegexSubtractionMethod import RegexSubtractionMethod
from semantic_metadata_extraction.methods.SameInputOutputMethod import SameInputOutputMethod


class SemanticMetadataExtraction:
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

    METHODS: list[Type[SemanticMethod]] = [
        SameInputOutputMethod,
        RegexMethod,
        RegexSubtractionMethod,
        DateParserWithBreaksMethod,
        DateParserMethod,
        MT5TrueCaseEnglishSpanishMethod,
    ]

    def __init__(self, extraction_identifier: ExtractionIdentifier):
        self.extraction_identifier = extraction_identifier

    def create_model(self, semantic_extraction_data: list[SemanticExtractionData]):
        if len(semantic_extraction_data) < 2:
            config_logger.info("\nBest method SameInputOutputMethod because no samples")
            return

        best_method_instance = self.get_best_method(semantic_extraction_data)
        best_method_instance.train(semantic_extraction_data)

    def get_best_method(self, semantic_extraction_data: list[SemanticExtractionData]):
        performance_semantic_extraction_data = [x for x in semantic_extraction_data if x.pdf_tags]

        best_performance = 0
        best_method_instance = self.METHODS[0](self.extraction_identifier)
        for method in self.METHODS[:-1]:
            method_instance = method(self.extraction_identifier)
            config_logger.info(f"\nChecking {method_instance.get_name()}")
            performance, _ = method_instance.performance(performance_semantic_extraction_data, 30)
            config_logger.info(f"\nPerformance {method_instance.get_name()}: {performance}%")
            if performance == 100:
                config_logger.info(f"\nBest method {method_instance.get_name()} with {performance}%")
                return method_instance

            if performance > best_performance:
                best_performance = performance
                best_method_instance = method_instance

        return self.decide_best_method_or_t5(best_performance, best_method_instance, performance_semantic_extraction_data)

    def decide_best_method_or_t5(
        self,
        best_performance: float,
        best_method_instance: SemanticMethod,
        semantic_extraction_data: list[SemanticExtractionData],
    ):
        if best_performance > 85:
            config_logger.info(f"\nBest method {best_method_instance.get_name()} with {best_performance}%")
            return best_method_instance

        t5 = MT5TrueCaseEnglishSpanishMethod(self.extraction_identifier)

        if best_performance < 60:
            config_logger.info(f"\nBest method {t5.get_name()} because the others were bad")
            return t5

        performance, _ = t5.performance(semantic_extraction_data, 30)
        config_logger.info(f"\nPerformance {t5.get_name()} with {performance}%")

        if performance > best_performance:
            config_logger.info(f"\nBest method {t5.get_name()} with {performance}%")
            return t5

        config_logger.info(f"\nBest method {best_method_instance.get_name()} with {best_performance}%")
        return best_method_instance

    def get_semantic_predictions(self, semantic_predictions_data: list[SemanticPredictionData]) -> list[str]:
        for method in self.METHODS:
            method_instance = method(self.extraction_identifier)
            method_path = join(self.extraction_identifier.get_path(), method_instance.get_name())
            config_logger.info(f"Checking {method_path}")

            if exists(method_path):
                config_logger.info(
                    f"Predicting {len(semantic_predictions_data)} documents with {method_instance.get_name()}"
                )
                return method_instance.predict(semantic_predictions_data)

        config_logger.info(f"Predicting {len(semantic_predictions_data)} documents with SameInputOutputMethod")
        return self.METHODS[0](self.extraction_identifier).predict(semantic_predictions_data)

    def remove_models(self):
        for method in self.METHODS:
            method_instance = method(self.extraction_identifier)
            method_instance.remove_model()
