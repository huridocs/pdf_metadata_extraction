
from rapidfuzz import fuzz

from metadata_extraction.PdfMetadata import PdfMetadata
from pdf_topic_classification.PdfLabels import PdfLabels
from pdf_topic_classification.PdfTopicClassificationMethod import PdfTopicClassificationMethod
from segment_selector.SegmentSelector import SegmentSelector

ratio_threshold = 75


class FuzzySegmentSelectorMethod(PdfTopicClassificationMethod):
    def get_all_appearance(self, pdf_metadata_segments: PdfMetadata) -> list[str]:
        appearances = []
        for option in self.options:
            for pdf_metadata_segment in pdf_metadata_segments.pdf_metadata_segments:
                if not pdf_metadata_segment.ml_label:
                    continue

                if fuzz.partial_ratio(option, pdf_metadata_segment.text_content) > ratio_threshold:
                    appearances.append(option)

        return list(set(appearances))

    @staticmethod
    def set_labeled_data(labels: list[str], pdf_metadata: PdfMetadata):
        for label in labels:
            for pdf_metadata_segment in pdf_metadata.pdf_metadata_segments:
                if fuzz.partial_ratio(label, pdf_metadata_segment.text_content) > ratio_threshold:
                    pdf_metadata_segment.ml_label = 1

    def predict(self, pdfs_labels: list[PdfLabels]):
        segment_selector = SegmentSelector(self.run_name, self.task_name)

        pdfs_metadata = list()
        for pdf_label in pdfs_labels:
            pdf_metadata = PdfMetadata(pdf_label.pdf_features, file_name=pdf_label.pdf_name)
            pdf_metadata.set_segments_from_paragraphs(pdf_label.paragraphs)
            pdfs_metadata.append(pdf_metadata)

        segment_selector.set_extraction_segments(pdfs_metadata)

        predictions = list()
        for pdf_metadata in pdfs_metadata:
            predictions.append(self.get_all_appearance(pdf_metadata))

        return predictions

    def train(self, pdfs_labels: list[PdfLabels]):
        pdfs_metadata_segments = list()

        for pdf_label in pdfs_labels:
            pdf_metadata = PdfMetadata(pdf_label.pdf_features, file_name=pdf_label.pdf_name)
            pdf_metadata.set_segments_from_paragraphs(pdf_label.paragraphs)
            self.set_labeled_data(pdf_label.labels, pdf_metadata)
            pdfs_metadata_segments.append(pdf_metadata)

        segment_selector = SegmentSelector(self.run_name, self.task_name)
        segment_selector.create_model(pdfs_metadata_segments)
