from paragraph_extraction_trainer.PdfSegment import PdfSegment
from rapidfuzz import fuzz

from pdf_topic_classification.PdfLabels import PdfLabels
from pdf_topic_classification.PdfTopicClassificationMethod import PdfTopicClassificationMethod


class FirstFuzzyCountry(PdfTopicClassificationMethod):
    @staticmethod
    def clean_text(text: str):
        text = text.lower()
        for to_remove in [
            "Republic of",
            "Bolivarian Republic of",
            "United",
            "Federative Republic of",
            "Eastern Republic of",
            "Plurinational State of",
            " Republic",
        ]:
            text = text.replace(to_remove.lower(), "")

        return text

    def get_first_appearance(self, pdf_segments: list[PdfSegment]) -> list[str]:
        for pdf_segment in pdf_segments:
            for ratio_threshold in range(100, 60, -10):
                for option in self.options:
                    if (
                        fuzz.partial_ratio(self.clean_text(option), self.clean_text(pdf_segment.text_content))
                        >= ratio_threshold
                    ):
                        return [option]

        return []

    def predict(self, pdfs_labels: list[PdfLabels]):
        predictions = list()
        for pdf_label in pdfs_labels:
            pdf_segments = [PdfSegment.from_pdf_tokens(x.tokens) for x in pdf_label.paragraphs]
            predictions.append(self.get_first_appearance(pdf_segments))

        return predictions

    def train(self, pdfs_labels: list[PdfLabels]):
        pass
