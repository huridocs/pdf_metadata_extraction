import json
import shutil
from os.path import join, exists
from pathlib import Path

from config import config_logger
from data.ExtractionIdentifier import ExtractionIdentifier
from data.Option import Option
from data.PredictionSample import PredictionSample
from data.Suggestion import Suggestion
from extractors.ExtractorBase import ExtractorBase
from extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import PdfMultiOptionMethod

from data.ExtractionData import ExtractionData
from data.TrainingSample import TrainingSample
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDigits3000 import (
    CleanBeginningDigits3000,
)
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDot1000 import CleanBeginningDot1000
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDot500 import CleanBeginningDot500
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanEndDot1000 import CleanEndDot1000
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitMethod import SetFitMethod
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SingleLabelSetFitMethod import SingleLabelSetFitMethod
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.TfIdfMethod import TfIdfMethod
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll100 import FuzzyAll100
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll75 import FuzzyAll75
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll88 import FuzzyAll88
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyFirst import FuzzyFirst
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyFirstCleanLabel import (
    FuzzyFirstCleanLabel,
)
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyLast import FuzzyLast
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyLastCleanLabel import FuzzyLastCleanLabel


class PdfToMultiOptionExtractor(ExtractorBase):

    MULTI_LABEL_METHODS: list[PdfMultiOptionMethod] = [
        FuzzyFirst(),
        FuzzyLast(),
        FuzzyFirstCleanLabel(),
        FuzzyLastCleanLabel(),
        FuzzyAll75(),
        FuzzyAll88(),
        FuzzyAll100(),
        PdfMultiOptionMethod(CleanBeginningDot500, SetFitMethod),
        PdfMultiOptionMethod(CleanEndDot1000, SetFitMethod),
    ]

    SINGLE_LABEL_METHODS: list[PdfMultiOptionMethod] = [
        FuzzyFirst(),
        FuzzyLast(),
        FuzzyFirstCleanLabel(),
        FuzzyLastCleanLabel(),
        FuzzyAll75(),
        FuzzyAll88(),
        FuzzyAll100(),
        PdfMultiOptionMethod(CleanBeginningDot500, SetFitMethod),
        PdfMultiOptionMethod(CleanBeginningDigits3000, TfIdfMethod),
        PdfMultiOptionMethod(CleanEndDot1000, TfIdfMethod),
        PdfMultiOptionMethod(CleanBeginningDot1000, SingleLabelSetFitMethod),
        PdfMultiOptionMethod(CleanEndDot1000, SingleLabelSetFitMethod),
    ]

    def __init__(self, extraction_identifier: ExtractionIdentifier):
        super().__init__(extraction_identifier)
        self.base_path = join(self.extraction_identifier.get_path(), "multi_option_extractor")
        self.options_path = join(self.base_path, "options.json")
        self.multi_value_path = join(self.base_path, "multi_value.json")
        self.method_name_path = Path(join(self.base_path, "method_name.json"))

        self.options: list[Option] = list()
        self.multi_value = False

    def create_model(self, extraction_data: ExtractionData):
        self.options = extraction_data.options
        self.multi_value = extraction_data.multi_value

        method = self.get_best_method(extraction_data)

        shutil.rmtree(self.base_path, ignore_errors=True)
        method.train(extraction_data)

        self.save_json(self.options_path, [x.model_dump() for x in extraction_data.options])
        self.save_json(self.multi_value_path, extraction_data.multi_value)
        self.save_json(str(self.method_name_path), method.get_name())

        return True, ""

    def get_suggestions(self, predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        if not predictions_samples:
            return []

        multi_option_samples, predictions = self.get_predictions(predictions_samples)

        suggestions = list()
        for multi_option_sample, prediction in zip(multi_option_samples, predictions):
            suggestion = Suggestion.get_empty(self.extraction_identifier, multi_option_sample.pdf_data.file_name)
            suggestion.add_prediction_multi_option(multi_option_sample, prediction)
            suggestions.append(suggestion)

        return suggestions

    def get_predictions(self, predictions_samples: list[PredictionSample]):
        self.load_options()
        multi_option_samples = [TrainingSample(pdf_data=sample.pdf_data) for sample in predictions_samples]
        multi_option_data = ExtractionData(
            multi_value=self.multi_value,
            options=self.options,
            samples=multi_option_samples,
            extraction_identifier=self.extraction_identifier,
        )
        method = self.get_predictions_method()
        method.set_parameters(multi_option_data)
        prediction = method.predict(multi_option_data)

        if not self.multi_value:
            prediction = [x[:1] for x in prediction]

        return multi_option_samples, prediction

    def load_options(self):
        if not exists(self.options_path) or not exists(self.multi_value_path):
            return

        with open(self.options_path, "r") as file:
            self.options = [Option(**x) for x in json.load(file)]

        with open(self.multi_value_path, "r") as file:
            self.multi_value = json.load(file)

    def get_best_method(self, multi_option_data: ExtractionData) -> PdfMultiOptionMethod:
        best_method_instance = self.SINGLE_LABEL_METHODS[0]
        best_performance = 0
        methods_to_loop = self.MULTI_LABEL_METHODS if self.multi_value else self.SINGLE_LABEL_METHODS
        for method in methods_to_loop:
            method.set_parameters(multi_option_data)
            if len(methods_to_loop) == 1:
                return method

            config_logger.info(f"\nChecking {method.get_name()}")

            try:
                performance = method.get_performance(multi_option_data)
            except Exception as e:
                config_logger.error(f"Error checking {method.get_name()}: {e}")
                performance = 0

            config_logger.info(f"\nPerformance {method.get_name()}: {performance}%")
            if performance == 100:
                config_logger.info(f"\nBest method {method.get_name()} with {performance}%")
                return method

            if performance > best_performance:
                best_performance = performance
                best_method_instance = method

        return best_method_instance

    def get_predictions_method(self):
        method_name = json.loads(self.method_name_path.read_text())
        for method in self.MULTI_LABEL_METHODS + self.SINGLE_LABEL_METHODS:
            if method.get_name() == method_name:
                return method

        return self.SINGLE_LABEL_METHODS[0]

    def is_valid(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.options:
            return False

        for sample in extraction_data.samples:
            if sample.pdf_data:
                return True

        return False
