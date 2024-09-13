import json
import os
from pathlib import Path

from config import config_logger
from data.ExtractionData import ExtractionData
from data.ExtractionIdentifier import ExtractionIdentifier
from data.PredictionSample import PredictionSample
from data.Suggestion import Suggestion
from extractors.ExtractorBase import ExtractorBase
from extractors.ToTextExtractorMethod import ToTextExtractorMethod
from extractors.text_to_text_extractor.methods.DateParserMethod import DateParserMethod
from extractors.text_to_text_extractor.methods.DateParserWithBreaksMethod import DateParserWithBreaksMethod
from extractors.text_to_text_extractor.methods.InputWithoutSpaces import InputWithoutSpaces
from extractors.text_to_text_extractor.methods.MT5TrueCaseEnglishSpanishMethod import MT5TrueCaseEnglishSpanishMethod
from extractors.text_to_text_extractor.methods.RegexMethod import RegexMethod
from extractors.text_to_text_extractor.methods.RegexSubtractionMethod import RegexSubtractionMethod
from extractors.text_to_text_extractor.methods.SameInputOutputMethod import SameInputOutputMethod
from send_logs import send_logs


class ToTextExtractor(ExtractorBase):
    METHODS: list[type[ToTextExtractorMethod]] = []

    def __init__(self, extraction_identifier: ExtractionIdentifier):
        super().__init__(extraction_identifier)
        extractor_path = self.extraction_identifier.get_path()
        self.method_name_path = Path(extractor_path, self.get_name(), "method_name.json")
        os.makedirs(self.method_name_path.parent, exist_ok=True)

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        pass

    def get_name(self):
        return self.__class__.__name__

    def get_suggestions(self, predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        method_instance = self.get_predictions_method()
        send_logs(
            self.extraction_identifier,
            f"And also using {method_instance.get_name()} to calculate {len(predictions_samples)} suggestions",
        )
        prediction = method_instance.predict(predictions_samples)
        suggestions = list()
        for prediction, prediction_sample in zip(prediction, predictions_samples):
            entity_name = prediction_sample.entity_name
            suggestions.append(Suggestion.from_prediction_text(self.extraction_identifier, entity_name, prediction))

        for suggestion, sample in zip(suggestions, predictions_samples):
            if sample.pdf_data:
                suggestion.add_segments(sample.pdf_data)

        return suggestions

    def get_predictions_method(self) -> ToTextExtractorMethod:
        if not self.method_name_path.exists():
            return self.METHODS[0](self.extraction_identifier)

        method_name = json.loads(self.method_name_path.read_text())
        for method in self.METHODS:
            method_instance = method(self.extraction_identifier)
            if method_instance.get_name() == method_name:
                return method_instance

        return self.METHODS[0](self.extraction_identifier)

    def create_model(self, extraction_data: ExtractionData) -> tuple[bool, str]:
        if not extraction_data.samples:
            return False, "No samples to create model"

        if len(extraction_data.samples) < 2:
            best_method_instance = self.METHODS[0](self.extraction_identifier)
            config_logger.info(f"\nBest method {best_method_instance.get_name()} because no samples")
            best_method_instance.train(extraction_data)
            return True, ""

        best_method_instance = self.get_best_method(extraction_data)
        self.method_name_path.write_text(json.dumps(best_method_instance.get_name()))
        best_method_instance.train(extraction_data)
        return True, ""

    def remove_models(self):
        for method in self.METHODS:
            method_instance = method(self.extraction_identifier)
            method_instance.remove_model()

    def get_best_method(self, extraction_data: ExtractionData):
        best_performance = 0
        best_method_instance = self.METHODS[0](self.extraction_identifier)
        for method in self.METHODS:
            method_instance = method(self.extraction_identifier)
            send_logs(self.extraction_identifier, f"Checking {method_instance.get_name()}")
            performance = method_instance.performance(extraction_data)
            send_logs(self.extraction_identifier, f"Performance {method_instance.get_name()}: {performance}%")
            if performance == 100:
                send_logs(self.extraction_identifier, f"Best method {method_instance.get_name()} with {performance}%")
                return method_instance

            if performance > best_performance:
                best_performance = performance
                best_method_instance = method_instance

        return best_method_instance
