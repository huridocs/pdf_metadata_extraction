from pdf_topic_classification.PdfLabels import PdfLabels
from pdf_topic_classification.PdfTopicClassificationMethod import PdfTopicClassificationMethod


class CheckData(PdfTopicClassificationMethod):
    def predict(self, pdfs_labels: list[PdfLabels]):
        return [[self.options[0]] for _ in pdfs_labels]

    def train(self, pdfs_labels: list[PdfLabels]):
        pass
