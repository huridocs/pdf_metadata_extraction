import json
from os.path import join, exists
from pathlib import Path

from data.ExtractionIdentifier import ExtractionIdentifier
from data.LogsMessage import Severity
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
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDotDigits500 import (
    CleanBeginningDotDigits500,
)
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanEndDotDigits1000 import CleanEndDotDigits1000
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.FastTextMethod import FastTextMethod
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitMethod import SetFitMethod
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SingleLabelSetFitMethod import SingleLabelSetFitMethod
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.TfIdfMethod import TfIdfMethod
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzy95 import (
    FastSegmentSelectorFuzzy95,
)
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzyCommas import (
    FastSegmentSelectorFuzzyCommas,
)
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll100 import FuzzyAll100
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll75 import FuzzyAll75
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyAll88 import FuzzyAll88
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyFirst import FuzzyFirst
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyFirstCleanLabel import (
    FuzzyFirstCleanLabel,
)
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyLast import FuzzyLast
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyLastCleanLabel import FuzzyLastCleanLabel
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzySegmentSelector import (
    FuzzySegmentSelector,
)
from send_logs import send_logs


class PdfToMultiOptionExtractor(ExtractorBase):
    METHODS: list[PdfMultiOptionMethod] = [
        FuzzyFirst(),
        FuzzyLast(),
        FuzzyFirstCleanLabel(),
        FuzzyLastCleanLabel(),
        FuzzyAll75(),
        FuzzyAll88(),
        FuzzyAll100(),
        FastSegmentSelectorFuzzy95(),
        FastSegmentSelectorFuzzyCommas(),
        FuzzySegmentSelector(),
        PdfMultiOptionMethod(CleanBeginningDigits3000, TfIdfMethod),
        PdfMultiOptionMethod(CleanEndDotDigits1000, TfIdfMethod),
        PdfMultiOptionMethod(CleanBeginningDotDigits500, FastTextMethod),
        PdfMultiOptionMethod(CleanEndDotDigits1000, FastTextMethod),
        PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitMethod),
        PdfMultiOptionMethod(CleanEndDotDigits1000, SetFitMethod),
        PdfMultiOptionMethod(CleanBeginningDotDigits500, SingleLabelSetFitMethod),
        PdfMultiOptionMethod(CleanEndDotDigits1000, SingleLabelSetFitMethod),
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
        method.train(extraction_data)

        self.save_json(self.options_path, [x.model_dump() for x in extraction_data.options])
        self.save_json(self.multi_value_path, extraction_data.multi_value)
        self.save_json(str(self.method_name_path), method.get_name())

        return True, ""

    def get_suggestions(self, predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        if not predictions_samples:
            return []

        training_samples, predictions = self.get_predictions(predictions_samples)
        prediction_method = self.get_predictions_method()

        context_from_the_end = "End" in prediction_method.get_name()
        suggestions = list()
        for training_sample, prediction_sample, prediction in zip(training_samples, predictions_samples, predictions):
            suggestion = Suggestion.get_empty(self.extraction_identifier, prediction_sample.entity_name)
            suggestion.add_prediction_multi_option(training_sample, prediction, context_from_the_end)
            suggestions.append(suggestion)

        return suggestions

    def get_predictions(self, predictions_samples: list[PredictionSample]) -> (list[TrainingSample], list[list[Option]]):
        self.load_options()
        training_samples = [TrainingSample(pdf_data=sample.pdf_data) for sample in predictions_samples]
        extraction_data = ExtractionData(
            multi_value=self.multi_value,
            options=self.options,
            samples=training_samples,
            extraction_identifier=self.extraction_identifier,
        )
        method = self.get_predictions_method()
        method.set_parameters(extraction_data)
        send_logs(self.extraction_identifier, f"Using method {method.get_name()} for suggestions")

        prediction = method.predict(extraction_data)

        if not self.multi_value:
            prediction = [x[:1] for x in prediction]

        return training_samples, prediction

    def load_options(self):
        if not exists(self.options_path) or not exists(self.multi_value_path):
            return

        with open(self.options_path, "r") as file:
            self.options = [Option(**x) for x in json.load(file)]

        with open(self.multi_value_path, "r") as file:
            self.multi_value = json.load(file)

    def get_best_method(self, multi_option_data: ExtractionData) -> PdfMultiOptionMethod:
        best_method_instance = self.METHODS[0]
        best_performance = 0
        for method in self.METHODS:
            performance = self.get_performance(method, multi_option_data)

            if performance == 100:
                send_logs(self.extraction_identifier, f"Best method {method.get_name()} with {performance}%")
                return method

            if performance > best_performance:
                best_performance = performance
                best_method_instance = method

        send_logs(self.extraction_identifier, f"Best method {best_method_instance.get_name()}")
        return best_method_instance

    def get_performance(self, method, multi_option_data):
        method.set_parameters(multi_option_data)

        if len(self.METHODS) == 1 or not method.can_be_used(multi_option_data):
            return 0

        send_logs(self.extraction_identifier, f"Checking {method.get_name()}")

        try:
            performance = method.get_performance(multi_option_data)
        except Exception as e:
            send_logs(self.extraction_identifier, f"Error checking {method.get_name()}: {e}", Severity.error)

            performance = 0

        send_logs(self.extraction_identifier, f"Performance {method.get_name()}: {performance}%")
        return performance

    def get_predictions_method(self):
        method_name = json.loads(self.method_name_path.read_text())
        for method in self.METHODS:
            if method.get_name() == method_name:
                return method

        return self.METHODS[0]

    def can_be_used(self, extraction_data: ExtractionData) -> bool:
        if not extraction_data.options:
            return False

        for sample in extraction_data.samples:
            if sample.pdf_data:
                return True

        return False
