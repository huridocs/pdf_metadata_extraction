from multi_option_extraction.PdfLabels import PdfLabels
from multi_option_extraction.PdfMultiOptionExtractionMethod import MultiOptionExtractionMethod


class NaiveMethod(MultiOptionExtractionMethod):
    def predict(self, multi_option_samples: list[PdfLabels]):
        return [[self.options[0]] for _ in multi_option_samples]

    def train(self, multi_option_samples: list[PdfLabels]):
        pass
