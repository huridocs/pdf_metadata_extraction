from multi_option_extraction.PdfLabels import PdfLabels
from multi_option_extraction.PdfMultiOptionExtractionMethod import PdfMultiOptionExtractionMethod


class NaiveMethod(PdfMultiOptionExtractionMethod):
    def predict(self, pdfs_labels: list[PdfLabels]):
        return [[self.options[0]] for _ in pdfs_labels]

    def train(self, pdfs_labels: list[PdfLabels]):
        pass
