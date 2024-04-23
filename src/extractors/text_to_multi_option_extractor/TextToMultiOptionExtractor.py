import json
import os
import shutil
from os.path import join, exists
from pathlib import Path
from typing import Type

from config import config_logger
from data.ExtractionData import ExtractionData
from data.Option import Option
from data.PredictionSample import PredictionSample
from data.Suggestion import Suggestion
from extractors.ExtractorBase import ExtractorBase
from extractors.text_to_multi_option_extractor.TextToMultiOptionMethod import TextToMultiOptionMethod
from extractors.text_to_multi_option_extractor.methods.TextBert import TextBert
from extractors.text_to_multi_option_extractor.methods.TextBertLarge import TextBertLarge
from extractors.text_to_multi_option_extractor.methods.TextFastTextMethod import TextFastTextMethod
from extractors.text_to_multi_option_extractor.methods.TextFuzzyAll100 import TextFuzzyAll100
from extractors.text_to_multi_option_extractor.methods.TextFuzzyAll75 import TextFuzzyAll75
from extractors.text_to_multi_option_extractor.methods.TextFuzzyAll88 import TextFuzzyAll88
from extractors.text_to_multi_option_extractor.methods.TextFuzzyFirst import TextFuzzyFirst
from extractors.text_to_multi_option_extractor.methods.NaiveTextToMultiOptionMethod import NaiveTextToMultiOptionMethod
from extractors.text_to_multi_option_extractor.methods.TextFuzzyFirstCleanLabels import TextFuzzyFirstCleanLabels
from extractors.text_to_multi_option_extractor.methods.TextFuzzyLast import TextFuzzyLast
from extractors.text_to_multi_option_extractor.methods.TextFuzzyLastCleanLabels import TextFuzzyLastCleanLabels
from extractors.text_to_multi_option_extractor.methods.TextSetFit import TextSetFit
from extractors.text_to_multi_option_extractor.methods.TextSetFitMultilingual import TextSetFitMultilingual
from extractors.text_to_multi_option_extractor.methods.TextSingleLabelBert import TextSingleLabelBert
from extractors.text_to_multi_option_extractor.methods.TextSingleLabelSetFit import TextSingleLabelSetFit
from extractors.text_to_multi_option_extractor.methods.TextSingleLabelSetFitMultilingual import (
    TextSingleLabelSetFitMultilingual,
)
from extractors.text_to_multi_option_extractor.methods.TextTfIdf import TextTfIdf


class TextToMultiOptionExtractor(ExtractorBase):
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

    METHODS: list[Type[TextToMultiOptionMethod]] = [
        # NaiveTextToMultiOptionMethod,
        # TextFuzzyAll75,
        # TextFuzzyAll88,
        # TextFuzzyAll100,
        # TextFuzzyFirst,
        # TextFuzzyFirstCleanLabels,
        # TextFuzzyLast,
        # TextFuzzyLastCleanLabels,
        # TextTfIdf,
        # TextFastTextMethod,
        # TextSetFit,
        # TextSetFitMultilingual,
        # TextSingleLabelSetFit,
        # TextSingleLabelSetFitMultilingual,
        # TextBert,
        TextSingleLabelBert,
    ]

    def __init__(self, extraction_identifier):
        super().__init__(extraction_identifier)

        self.base_path = join(self.extraction_identifier.get_path(), "text_to_multi_option")
        self.options_path = join(self.base_path, "options.json")
        self.multi_value_path = join(self.base_path, "multi_value.json")
        self.method_name_path = Path(join(self.base_path, "method_name.json"))

        self.options: list[Option] = list()
        self.multi_value = False

    def get_suggestions(self, predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        if not predictions_samples:
            return []

        predictions = self.get_predictions_method().predict(predictions_samples)

        if not self.multi_value:
            predictions = [x[:1] for x in predictions]

        suggestions = list()
        for prediction_sample, prediction in zip(predictions_samples, predictions):
            suggestion = Suggestion.from_prediction_multi_option(
                self.extraction_identifier, prediction_sample.entity_name, prediction
            )
            suggestions.append(suggestion)

        return suggestions

    def get_predictions_method(self):
        self.load_options()
        method_name = json.loads(self.method_name_path.read_text())
        for method in self.METHODS:
            method_instance = method(self.extraction_identifier, self.options, self.multi_value)
            if method_instance.get_name() == method_name:
                return method_instance

        return self.METHODS[0](self.extraction_identifier, self.options, self.multi_value)

    def load_options(self):
        if not exists(self.options_path) or not exists(self.multi_value_path):
            return

        with open(self.options_path, "r") as file:
            self.options = [Option(**x) for x in json.load(file)]

        with open(self.multi_value_path, "r") as file:
            self.multi_value = json.load(file)

    def create_model(self, extraction_data: ExtractionData) -> tuple[bool, str]:
        self.options = extraction_data.options
        self.multi_value = extraction_data.multi_value

        shutil.rmtree(self.extraction_identifier.get_path(), ignore_errors=True)

        best_method_instance = self.get_best_method(extraction_data)
        best_method_instance.train(extraction_data)

        self.save_json(self.options_path, [x.model_dump() for x in extraction_data.options])
        self.save_json(self.multi_value_path, extraction_data.multi_value)
        self.save_json(str(self.method_name_path), best_method_instance.get_name())
        return True, ""

    def get_best_method(self, extraction_data: ExtractionData):
        best_performance = 0
        best_method_instance = self.METHODS[0](self.extraction_identifier, self.options, self.multi_value)
        for method in self.METHODS:
            method_instance = method(self.extraction_identifier, self.options, self.multi_value)

            if len(self.METHODS) == 1:
                return method_instance

            if not method_instance.can_be_used(extraction_data):
                continue

            performance = self.get_performance(extraction_data, method_instance)
            if performance == 100:
                config_logger.info(f"\nBest method {method_instance.get_name()} with {performance}%")
                return method_instance

            if performance > best_performance:
                best_performance = performance
                best_method_instance = method_instance

        return best_method_instance

    @staticmethod
    def get_performance(extraction_data, method_instance):
        config_logger.info(f"\nChecking {method_instance.get_name()}")
        try:
            performance = method_instance.performance(extraction_data)
        except:
            performance = 0
        config_logger.info(f"\nPerformance {method_instance.get_name()}: {performance}%")
        return performance

    def remove_models(self):
        for method in self.METHODS:
            method_instance = method(self.extraction_identifier, self.options, self.multi_value)
            method_instance.remove_model()

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.options:
            return False

        for sample in extraction_data.samples:
            if sample.labeled_data and sample.labeled_data.source_text:
                return True

        return False

    def suggestions_from_predictions(
        self, method_instance: type[TextToMultiOptionMethod], predictions_samples: list[PredictionSample]
    ) -> list[Suggestion]:
        suggestions = list()
        prediction = method_instance.predict(predictions_samples, self.options)

        for prediction, prediction_sample in zip(prediction, predictions_samples):
            suggestion = Suggestion.from_prediction_multi_option(
                self.extraction_identifier, prediction_sample.entity_name, prediction
            )
            suggestions.append(suggestion)

        return suggestions
