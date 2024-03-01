
from paragraph_extraction_trainer.PdfSegment import PdfSegment
from rapidfuzz import fuzz

from pdf_topic_classification.PdfLabels import PdfLabels
from pdf_topic_classification.PdfTopicClassificationMethod import PdfTopicClassificationMethod

ratio_threshold = 88


class All88FuzzyMethod(PdfTopicClassificationMethod):
    def get_all_appearance(self, pdf_segments: list[PdfSegment]) -> list[str]:
        appearances = []
        for option in self.options:
            for pdf_segment in pdf_segments:
                ratio = fuzz.partial_ratio(option.lower(), pdf_segment.text_content.lower())
                if ratio >= ratio_threshold:
                    appearances.append(option)

        return list(set(appearances))

    def predict(self, pdfs_labels: list[PdfLabels]):
        predictions = list()
        for pdf_label in pdfs_labels:
            pdf_segments = [PdfSegment.from_pdf_tokens(x.tokens) for x in pdf_label.paragraphs]
            predictions.append(self.get_all_appearance(pdf_segments))

        return predictions

    def train(self, pdfs_labels: list[PdfLabels]):
        pass