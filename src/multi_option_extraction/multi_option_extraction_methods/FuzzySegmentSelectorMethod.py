from rapidfuzz import fuzz

from metadata_extraction.PdfData import PdfData
from multi_option_extraction.PdfLabels import PdfLabels
from multi_option_extraction.PdfMultiOptionExtractionMethod import MultiOptionExtractionMethod
from segment_selector.SegmentSelector import SegmentSelector

ratio_threshold = 75


class FuzzySegmentSelectorMethod(MultiOptionExtractionMethod):
    def get_all_appearance(self, pdf_metadata_segments: PdfData) -> list[str]:
        appearances = []
        for option in self.options:
            for pdf_metadata_segment in pdf_metadata_segments.pdf_data_segments:
                if not pdf_metadata_segment.ml_label:
                    continue

                if fuzz.partial_ratio(option.lower(), pdf_metadata_segment.text_content.lower()) > ratio_threshold:
                    appearances.append(option)

        return list(set(appearances))

    @staticmethod
    def set_labeled_data(labels: list[str], pdf_metadata: PdfData):
        for label in labels:
            for pdf_metadata_segment in pdf_metadata.pdf_data_segments:
                if fuzz.partial_ratio(label, pdf_metadata_segment.text_content) > ratio_threshold:
                    pdf_metadata_segment.ml_label = 1

    def predict(self, multi_option_samples: list[PdfLabels]):
        segment_selector = SegmentSelector(self.tenant, self.extraction_id)

        pdfs_metadata = list()
        for pdf_label in multi_option_samples:
            pdf_metadata = PdfData(pdf_label.pdf_features, file_name=pdf_label.pdf_name)
            pdf_metadata.set_segments_from_paragraphs(pdf_label.paragraphs)
            pdfs_metadata.append(pdf_metadata)

        segment_selector.set_extraction_segments(pdfs_metadata)

        predictions = list()
        for pdf_metadata in pdfs_metadata:
            predictions.append(self.get_all_appearance(pdf_metadata))

        return predictions

    def train(self, multi_option_samples: list[PdfLabels]):
        pdfs_metadata_segments = list()

        for pdf_label in multi_option_samples:
            pdf_metadata = PdfData(pdf_label.pdf_features, file_name=pdf_label.pdf_name)
            pdf_metadata.set_segments_from_paragraphs(pdf_label.paragraphs)
            self.set_labeled_data(pdf_label.labels, pdf_metadata)
            pdfs_metadata_segments.append(pdf_metadata)

        segment_selector = SegmentSelector(self.tenant, self.extraction_id)
        segment_selector.create_model(pdfs_metadata_segments)
