import json
import os
from os import makedirs
from os.path import join, exists
from pathlib import Path

from config import config_logger
from data.ExtractionIdentifier import ExtractionIdentifier
from data.Option import Option
from metadata_extraction.PdfData import PdfData
from multi_option_extraction.MultiOptionExtractionMethod import MultiOptionExtractionMethod
from multi_option_extraction.data.MultiOptionData import MultiOptionData
from multi_option_extraction.data.MultiOptionSample import MultiOptionSample
from multi_option_extraction.filter_segments_methods.CleanBeginningDigits3000 import CleanBeginningDigits3000
from multi_option_extraction.filter_segments_methods.CleanBeginningDot1000 import CleanBeginningDot1000
from multi_option_extraction.filter_segments_methods.CleanBeginningDot250 import CleanBeginningDot250
from multi_option_extraction.filter_segments_methods.CleanEndDot1000 import CleanEndDot1000
from multi_option_extraction.filter_segments_methods.CleanEndDot250 import CleanEndDot250
from multi_option_extraction.multi_labels_methods.BertBatch1 import BertBatch1
from multi_option_extraction.multi_labels_methods.SingleLabelBert import SingleLabelBert
from multi_option_extraction.multi_labels_methods.TfIdfMethod import TfIdfMethod
from multi_option_extraction.multi_option_extraction_methods.FuzzyAll100 import FuzzyAll100
from multi_option_extraction.multi_option_extraction_methods.FuzzyAll75 import FuzzyAll75
from multi_option_extraction.multi_option_extraction_methods.FuzzyAll88 import FuzzyAll88
from multi_option_extraction.multi_option_extraction_methods.FuzzyFirst import FuzzyFirst
from multi_option_extraction.multi_option_extraction_methods.FuzzyFirstCleanLabel import FuzzyFirstCleanLabel
from multi_option_extraction.multi_option_extraction_methods.FuzzyLast import FuzzyLast
from multi_option_extraction.multi_option_extraction_methods.FuzzyLastCleanLabel import FuzzyLastCleanLabel


class MultiOptionExtractor:

    MULTI_LABEL_METHODS: list[MultiOptionExtractionMethod] = [
        FuzzyFirst(),
        FuzzyLast(),
        FuzzyFirstCleanLabel(),
        FuzzyLastCleanLabel(),
        FuzzyAll75(),
        FuzzyAll88(),
        FuzzyAll100(),
        MultiOptionExtractionMethod(CleanBeginningDigits3000, TfIdfMethod),
        MultiOptionExtractionMethod(CleanBeginningDot250, BertBatch1),
        MultiOptionExtractionMethod(CleanEndDot250, BertBatch1),
        MultiOptionExtractionMethod(CleanBeginningDot1000, BertBatch1),
        MultiOptionExtractionMethod(CleanEndDot1000, BertBatch1),
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
        MultiOptionExtractionMethod(CleanBeginningDot250, SingleLabelBert),
        MultiOptionExtractionMethod(CleanEndDot250, SingleLabelBert),
        MultiOptionExtractionMethod(CleanBeginningDot1000, SingleLabelBert),
        MultiOptionExtractionMethod(CleanEndDot1000, SingleLabelBert),
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

    def create_model(self, multi_option_data: MultiOptionData):
        self.options = multi_option_data.options
        self.multi_value = multi_option_data.multi_value

        self.save_json(self.options_path, [x.model_dump() for x in multi_option_data.options])
        self.save_json(self.multi_value_path, multi_option_data.multi_value)

        method = self.get_best_method(multi_option_data)
        method.train(multi_option_data)

        os.makedirs(self.method_name_path.parent, exist_ok=True)
        self.method_name_path.write_text(method.get_name())

        return True, ""

    @staticmethod
    def save_json(path: str, data: any):
        if not exists(Path(path).parent):
            makedirs(Path(path).parent)

        with open(path, "w") as file:
            json.dump(data, file)

    def get_multi_option_predictions(self, pdfs_data: list[PdfData]) -> list[MultiOptionSample]:
        self.load_options()
        multi_option_samples = [MultiOptionSample(pdf_data=pdf_data) for pdf_data in pdfs_data]
        multi_option_data = MultiOptionData(
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

        for prediction, multi_option_sample in zip(prediction, multi_option_samples):
            multi_option_sample.values = prediction

        return multi_option_samples

    def load_options(self):
        if not exists(self.options_path) or not exists(self.multi_value_path):
            return

        with open(self.options_path, "r") as file:
            self.options = [Option(**x) for x in json.load(file)]

        with open(self.multi_value_path, "r") as file:
            self.multi_value = json.load(file)

    def get_best_method(self, multi_option_data: MultiOptionData) -> MultiOptionExtractionMethod:
        best_method_instance = self.SINGLE_LABEL_METHODS[0]
        best_performance = 0
        methods_to_loop = self.MULTI_LABEL_METHODS if self.multi_value else self.SINGLE_LABEL_METHODS
        for method in methods_to_loop:
            method.set_parameters(multi_option_data)
            config_logger.info(f"\nChecking {method.get_name()}")
            performance = method.get_performance(multi_option_data)
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
        multi_option_extractor = MultiOptionExtractor(extraction_identifier)
        multi_option_extractor.load_options()
        return len(multi_option_extractor.options) > 0

    def get_predictions_method(self):
        method_name = self.method_name_path.read_text()
        for method in self.MULTI_LABEL_METHODS + self.SINGLE_LABEL_METHODS:
            if method.get_name() == method_name:
                return method

        return self.SINGLE_LABEL_METHODS[0]
