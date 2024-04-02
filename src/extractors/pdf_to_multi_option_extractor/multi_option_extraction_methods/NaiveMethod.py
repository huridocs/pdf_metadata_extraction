from extractors.pdf_to_multi_option_extractor.MultiOptionExtractionMethod import MultiOptionExtractionMethod
from data.ExtractionData import ExtractionData


class NaiveMethod(MultiOptionExtractionMethod):
    def predict(self, multi_option_data: ExtractionData):
        return [multi_option_data.options[0] for _ in multi_option_data.samples]

    def train(self, multi_option_data: ExtractionData):
        pass
