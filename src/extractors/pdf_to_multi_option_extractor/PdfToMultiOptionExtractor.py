import json
import os
import shutil
from os import makedirs
from os.path import join, exists
from pathlib import Path

from config import config_logger
from data.ExtractionIdentifier import ExtractionIdentifier
from data.Option import Option
from data.Suggestion import Suggestion
from data.PdfData import PdfData
from extractors.pdf_to_multi_option_extractor.MultiOptionExtractionMethod import MultiOptionExtractionMethod
from data.ExtractionData import ExtractionData
from data.TrainingSample import TrainingSample
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDigits3000 import (
    CleanBeginningDigits3000,
)
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDot1000 import CleanBeginningDot1000
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDot250 import CleanBeginningDot250
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanEndDot1000 import CleanEndDot1000
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanEndDot250 import CleanEndDot250
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.BertBatch1 import BertBatch1
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SingleLabelBert import SingleLabelBert
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


class PdfToMultiOptionExtractor:
    MULTI_LABEL_METHODS: list[MultiOptionExtractionMethod] = [
        FuzzyFirst(),
        FuzzyLast(),
        FuzzyFirstCleanLabel(),
        FuzzyLastCleanLabel(),
        FuzzyAll75(),
        FuzzyAll88(),
        FuzzyAll100(),
        MultiOptionExtractionMethod(CleanBeginningDigits3000, TfIdfMethod),
        MultiOptionExtractionMethod(CleanEndDot1000, TfIdfMethod),
        MultiOptionExtractionMethod(CleanBeginningDot250, BertBatch1),
        MultiOptionExtractionMethod(CleanEndDot250, BertBatch1),
        MultiOptionExtractionMethod(CleanBeginningDot1000, BertBatch1),
        MultiOptionExtractionMethod(CleanEndDot1000, BertBatch1),
    ]

    SINGLE_LABEL_METHODS: list[MultiOptionExtractionMethod] = [
        FuzzyFirst(),
        FuzzyLast(),
        FuzzyFirstCleanLabel(),
        FuzzyLastCleanLabel(),
        FuzzyAll75(),
        FuzzyAll88(),
        FuzzyAll100(),
        MultiOptionExtractionMethod(CleanBeginningDigits3000, TfIdfMethod),
        MultiOptionExtractionMethod(CleanEndDot1000, TfIdfMethod),
        MultiOptionExtractionMethod(CleanBeginningDot250, SingleLabelBert),
        MultiOptionExtractionMethod(CleanEndDot250, SingleLabelBert),
        MultiOptionExtractionMethod(CleanBeginningDot1000, SingleLabelBert),
        MultiOptionExtractionMethod(CleanEndDot1000, SingleLabelBert),
    ]

    def __init__(self, extraction_identifier: ExtractionIdentifier):
        self.extraction_identifier = extraction_identifier

        self.base_path = join(self.extraction_identifier.get_path(), "multi_option_extractor")
        self.options_path = join(self.base_path, "options.json")
        self.multi_value_path = join(self.base_path, "multi_value.json")
        self.method_name_path = Path(join(self.base_path, "method_name.json"))

        self.options: list[Option] = list()
        self.multi_value = False

    def create_model(self, multi_option_data: ExtractionData):
        self.options = multi_option_data.options
        self.multi_value = multi_option_data.multi_value

        method = self.get_best_method(multi_option_data)

        shutil.rmtree(self.base_path, ignore_errors=True)
        method.train(multi_option_data)

        os.makedirs(self.method_name_path.parent, exist_ok=True)
        self.save_json(self.options_path, [x.model_dump() for x in multi_option_data.options])
        self.save_json(self.multi_value_path, multi_option_data.multi_value)
        self.method_name_path.write_text(method.get_name())

        return True, ""

    @staticmethod
    def save_json(path: str, data: any):
        if not exists(Path(path).parent):
            makedirs(Path(path).parent)

        with open(path, "w") as file:
            json.dump(data, file)

    def get_suggestions(self, pdfs_data: list[PdfData]) -> list[Suggestion]:
        if not pdfs_data:
            return []

        multi_option_samples, predictions = self.get_predictions(pdfs_data)

        suggestions = list()
        for multi_option_sample, prediction in zip(multi_option_samples, predictions):
            suggestion = Suggestion.get_empty(self.extraction_identifier, multi_option_sample.pdf_data.file_name)
            suggestion.add_prediction_multi_option(multi_option_sample, prediction)
            suggestions.append(suggestion)

        return suggestions

    def get_predictions(self, pdfs_data):
        self.load_options()
        multi_option_samples = [TrainingSample(pdf_data=pdf_data) for pdf_data in pdfs_data]
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

    def get_best_method(self, multi_option_data: ExtractionData) -> MultiOptionExtractionMethod:
        best_method_instance = self.SINGLE_LABEL_METHODS[0]
        best_performance = 0
        methods_to_loop = self.MULTI_LABEL_METHODS if self.multi_value else self.SINGLE_LABEL_METHODS
        for method in methods_to_loop:
            method.set_parameters(multi_option_data)
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

    @staticmethod
    def is_multi_option_extraction(extraction_identifier: ExtractionIdentifier):
        multi_option_extractor = PdfToMultiOptionExtractor(extraction_identifier)
        multi_option_extractor.load_options()
        return len(multi_option_extractor.options) > 0

    def get_predictions_method(self):
        method_name = self.method_name_path.read_text()
        for method in self.MULTI_LABEL_METHODS + self.SINGLE_LABEL_METHODS:
            if method.get_name() == method_name:
                return method

        return self.SINGLE_LABEL_METHODS[0]
