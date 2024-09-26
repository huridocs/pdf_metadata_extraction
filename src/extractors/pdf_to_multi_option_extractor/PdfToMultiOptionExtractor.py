import json
import shutil
from collections import Counter
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
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDot1000 import CleanBeginningDot1000
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDotDigits500 import (
    CleanBeginningDotDigits500,
)
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanEndDotDigits1000 import CleanEndDotDigits1000
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.FastTextMethod import FastTextMethod
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitEnglishMethod import SetFitEnglishMethod
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitMultilingualMethod import SetFitMultilingualMethod
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SingleLabelSetFitEnglishMethod import (
    SingleLabelSetFitEnglishMethod,
)
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SingleLabelSetFitMultilingualMethod import (
    SingleLabelSetFitMultilingualMethod,
)
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
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.NextWordsTokenSelectorFuzzy75 import (
    NextWordsTokenSelectorFuzzy75,
)

from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.PreviousWordsSentenceSelectorFuzzyCommas import (
    PreviousWordsSentenceSelectorFuzzyCommas,
)
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.PreviousWordsTokenSelectorFuzzy75 import (
    PreviousWordsTokenSelectorFuzzy75,
)
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.SentenceSelectorFuzzyCommas import (
    SentenceSelectorFuzzyCommas,
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
        PreviousWordsTokenSelectorFuzzy75(),
        NextWordsTokenSelectorFuzzy75(),
        PreviousWordsSentenceSelectorFuzzyCommas(),
        SentenceSelectorFuzzyCommas(),
        FastSegmentSelectorFuzzy95(),
        FastSegmentSelectorFuzzyCommas(),
        FuzzySegmentSelector(),
        PdfMultiOptionMethod(CleanBeginningDotDigits500, FastTextMethod),
        PdfMultiOptionMethod(CleanEndDotDigits1000, FastTextMethod),
        PdfMultiOptionMethod(CleanBeginningDot1000, SetFitEnglishMethod),
        PdfMultiOptionMethod(CleanBeginningDot1000, SetFitMultilingualMethod),
        PdfMultiOptionMethod(CleanBeginningDot1000, SingleLabelSetFitEnglishMethod),
        PdfMultiOptionMethod(CleanBeginningDot1000, SingleLabelSetFitMultilingualMethod),
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

        send_logs(self.extraction_identifier, self.get_stats(extraction_data))

        performance_train_set, performance_test_set = ExtractorBase.get_train_test_sets(extraction_data)
        send_logs(self.extraction_identifier, f"Train set contains {len(performance_train_set.samples)} samples")
        send_logs(self.extraction_identifier, f"Test set contains {len(performance_test_set.samples)} samples")

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

        return method.get_samples_for_context(extraction_data), prediction

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
        performance_log = "Performance aggregation:\n"
        for method in self.METHODS:
            performance = self.get_method_performance(method, multi_option_data)
            performance_log += f"{method.get_name()}: {round(performance, 2)}%\n"
            if performance == 100:
                send_logs(self.extraction_identifier, performance_log)
                send_logs(self.extraction_identifier, f"Best method {method.get_name()} with {performance}%")
                return method

            if performance > best_performance:
                best_performance = performance
                best_method_instance = method

        send_logs(self.extraction_identifier, performance_log)
        send_logs(self.extraction_identifier, f"Best method {best_method_instance.get_name()}")
        return best_method_instance

    def get_method_performance(self, method: PdfMultiOptionMethod, multi_option_data: ExtractionData) -> float:
        method.set_parameters(multi_option_data)

        if not method.can_be_used(multi_option_data):
            send_logs(self.extraction_identifier, f"Not valid method {method.get_name()}")
            return 0

        send_logs(self.extraction_identifier, f"Checking {method.get_name()}")

        try:
            performance = method.get_performance(multi_option_data)
        except Exception as e:
            severity = Severity.error if method.REPORT_ERRORS else Severity.info
            send_logs(self.extraction_identifier, f"Error checking {method.get_name()}: {e}", severity)
            performance = 0

        self.reset_extraction_data(multi_option_data)

        if method.multi_label_method:
            shutil.rmtree(method.base_path, ignore_errors=True)

        send_logs(self.extraction_identifier, f"Performance {method.get_name()}: {round(performance, 2)}%")
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

    @staticmethod
    def reset_extraction_data(multi_option_data: ExtractionData):
        for sample in multi_option_data.samples:
            for segment in sample.pdf_data.pdf_data_segments:
                segment.ml_label = 0

    @staticmethod
    def get_stats(extraction_data: ExtractionData):
        options = Counter()
        for sample in extraction_data.samples:
            options.update([option.label for option in sample.labeled_data.values])
        languages = Counter()
        for sample in extraction_data.samples:
            languages.update([sample.labeled_data.language_iso])

        options_count = len(extraction_data.options)
        stats = f"\nNumber of options: {options_count}\n"
        stats += f"Number of samples: {len(extraction_data.samples)}\n"
        stats += f"Languages\n"
        stats += "\n".join([f"{key} {value}" for key, value in languages.most_common()])
        stats += f"\nOptions\n"
        stats += "\n".join([f"{key} {value}" for key, value in options.most_common()])
        return stats
