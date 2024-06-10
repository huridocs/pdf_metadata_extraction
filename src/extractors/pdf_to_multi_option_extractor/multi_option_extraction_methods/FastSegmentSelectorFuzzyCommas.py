from data.Option import Option
from data.ExtractionData import ExtractionData
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzy95 import (
    FastSegmentSelectorFuzzy95,
)
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FuzzyCommas import FuzzyCommas


class FastSegmentSelectorFuzzyCommas(FastSegmentSelectorFuzzy95):

    def predict(self, multi_option_data: ExtractionData) -> list[list[Option]]:
        predict_data = self.get_prediction_data(multi_option_data)
        return FuzzyCommas().predict(predict_data)
