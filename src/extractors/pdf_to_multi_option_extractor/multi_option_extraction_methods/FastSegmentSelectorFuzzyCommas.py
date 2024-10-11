from data.Option import Option
from data.ExtractionData import ExtractionData
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzy95 import (
    FastSegmentSelectorFuzzy95,
)
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyCommas import FuzzyCommas


class FastSegmentSelectorFuzzyCommas(FastSegmentSelectorFuzzy95):
    REPORT_ERRORS = False

    def predict(self, multi_option_data: ExtractionData) -> list[list[Option]]:
        self.set_parameters(multi_option_data)
        self.extraction_data = self.get_prediction_data(multi_option_data)
        return FuzzyCommas().predict(self.extraction_data)

    def train(self, multi_option_data: ExtractionData):
        self.set_parameters(multi_option_data)
        super().train(multi_option_data)
        FuzzyCommas().train(multi_option_data)
