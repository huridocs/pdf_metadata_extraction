from extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import PdfMultiOptionMethod

from data.ExtractionData import ExtractionData


class NaiveMethod(PdfMultiOptionMethod):
    def predict(self, multi_option_data: ExtractionData):
        return [multi_option_data.options[0] for _ in multi_option_data.samples]

    def train(self, multi_option_data: ExtractionData):
        pass
