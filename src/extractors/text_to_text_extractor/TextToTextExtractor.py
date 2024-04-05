import os
from os.path import exists, join
from typing import Type

from config import config_logger
from data.ExtractionData import ExtractionData
from data.PredictionSample import PredictionSample
from data.Suggestion import Suggestion
from extractors.ExtractorBase import ExtractorBase
from extractors.text_to_text_extractor.TextToTextMethod import TextToTextMethod
from extractors.text_to_text_extractor.methods.DateParserMethod import DateParserMethod
from extractors.text_to_text_extractor.methods.DateParserWithBreaksMethod import DateParserWithBreaksMethod
from extractors.text_to_text_extractor.methods.MT5TrueCaseEnglishSpanishMethod import MT5TrueCaseEnglishSpanishMethod
from extractors.text_to_text_extractor.methods.RegexMethod import RegexMethod
from extractors.text_to_text_extractor.methods.RegexSubtractionMethod import RegexSubtractionMethod
from extractors.text_to_text_extractor.methods.SameInputOutputMethod import SameInputOutputMethod


class TextToTextExtractor(ExtractorBase):
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

    METHODS: list[Type[TextToTextMethod]] = [
        SameInputOutputMethod,
        RegexMethod,
        RegexSubtractionMethod,
        DateParserWithBreaksMethod,
        DateParserMethod,
        MT5TrueCaseEnglishSpanishMethod,
    ]

    def create_model(self, extraction_data: ExtractionData) -> tuple[bool, str]:
        if len(extraction_data.samples) < 2:
            config_logger.info("\nBest method SameInputOutputMethod because no samples")
            return True, ""

        best_method_instance = self.get_best_method(extraction_data)
        best_method_instance.train(extraction_data)
        return True, ""

    def get_best_method(self, extraction_data: ExtractionData):
        best_performance = 0
        best_method_instance = self.METHODS[0](self.extraction_identifier)
        for method in self.METHODS[:-1]:
            method_instance = method(self.extraction_identifier)
            config_logger.info(f"\nChecking {method_instance.get_name()}")
            performance, _ = method_instance.performance(extraction_data)
            config_logger.info(f"\nPerformance {method_instance.get_name()}: {performance}%")
            if performance == 100:
                config_logger.info(f"\nBest method {method_instance.get_name()} with {performance}%")
                return method_instance

            if performance > best_performance:
                best_performance = performance
                best_method_instance = method_instance

        return self.decide_best_method_or_t5(best_performance, best_method_instance, extraction_data)

    def decide_best_method_or_t5(
        self,
        best_performance: float,
        best_method_instance: TextToTextMethod,
        extraction_data: ExtractionData,
    ):
        if best_performance > 85:
            config_logger.info(f"\nBest method {best_method_instance.get_name()} with {best_performance}%")
            return best_method_instance

        t5 = MT5TrueCaseEnglishSpanishMethod(self.extraction_identifier)

        if best_performance < 60:
            config_logger.info(f"\nBest method {t5.get_name()} because the others were bad")
            return t5

        performance, _ = t5.performance(extraction_data)
        config_logger.info(f"\nPerformance {t5.get_name()} with {performance}%")

        if performance > best_performance:
            config_logger.info(f"\nBest method {t5.get_name()} with {performance}%")
            return t5

        config_logger.info(f"\nBest method {best_method_instance.get_name()} with {best_performance}%")
        return best_method_instance

    def get_suggestions(self, predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        for method in self.METHODS:
            method_instance = method(self.extraction_identifier)
            method_path = join(self.extraction_identifier.get_path(), method_instance.get_name())
            config_logger.info(f"Checking {method_path}")

            if exists(method_path):
                config_logger.info(
                    f"Predicting {len(predictions_samples)} documents with {method_instance.get_name()}"
                )
                return self.suggestions_from_predictions(method_instance, predictions_samples)

        config_logger.info(f"Predicting {len(predictions_samples)} documents with SameInputOutputMethod")
        naive_method = self.METHODS[0](self.extraction_identifier)
        return self.suggestions_from_predictions(naive_method, predictions_samples)

    def suggestions_from_predictions(self, method_instance: type[TextToTextMethod], predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        suggestions = list()
        prediction = method_instance.predict(predictions_samples)

        for prediction, prediction_sample in zip(prediction, predictions_samples):
            suggestion = Suggestion.from_prediction_text(
                self.extraction_identifier, prediction_sample.entity_name , prediction
            )
            suggestions.append(suggestion)

        return suggestions

    def remove_models(self):
        for method in self.METHODS:
            method_instance = method(self.extraction_identifier)
            method_instance.remove_model()

    def is_valid(self, extraction_data: ExtractionData) -> bool:
        for sample in extraction_data.samples:
            if sample.tags_texts:
                return True

        return True
