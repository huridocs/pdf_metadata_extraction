from time import time

from langcodes import standardize_tag

from config import config_logger
from data.ExtractionIdentifier import ExtractionIdentifier
from data.LabeledData import LabeledData
from data.PdfTagData import PdfTagData
from data.SemanticExtractionData import SemanticExtractionData
from data.SemanticPredictionData import SemanticPredictionData
from data.PdfData import PdfData
from data.PdfDataSegment import PdfDataSegment
from extractors.segment_selector.SegmentSelector import SegmentSelector
from extractors.text_to_text_extractor.TextToTextExtractor import TextToTextExtractor


class PdfToTextExtractor:
    def __init__(self, extraction_identifier: ExtractionIdentifier, pdfs_data: list[PdfData]):
        self.extraction_identifier = extraction_identifier
        self.pdfs_data = pdfs_data
        self.segment_selector = SegmentSelector(self.extraction_identifier)

    def create_model(self, labeled_data: list[LabeledData]) -> (bool, str):
        all_pdf_segments: list[PdfDataSegment] = [
            pdf_segment for pdf_segments in self.pdfs_data for pdf_segment in pdf_segments.pdf_data_segments
        ]

        if not all_pdf_segments:
            return False, "No data to create model"

        segment_selector = SegmentSelector(self.extraction_identifier)
        segment_selector.create_model(pdfs_data=self.pdfs_data)

        self.create_semantic_model(labeled_data)
        return True, ""

    def create_semantic_model(self, labeled_data: list[LabeledData]):
        semantic_metadata_extraction = TextToTextExtractor(self.extraction_identifier)
        semantic_metadata_extraction.remove_models()
        semantic_extractions_data: list[SemanticExtractionData] = list()
        for pdf_features, labeled_data in zip(self.pdfs_data, labeled_data):
            semantic_extraction_data = SemanticExtractionData(
                text=labeled_data.label_text.strip(),
                pdf_tags=(self.get_predicted_tags_data(pdf_features)),
                language_iso=standardize_tag(labeled_data.language_iso),
            )

            semantic_extractions_data.append(semantic_extraction_data)

        semantic_metadata_extraction.create_model(semantic_extractions_data)

    @staticmethod
    def get_predicted_tags_data(pdf_data: PdfData) -> list[PdfTagData]:
        predicted_pdf_segments = [x for x in pdf_data.pdf_data_segments if x.ml_label]

        pdfs_tag_data: list[PdfTagData] = list()
        for pdf_segment in predicted_pdf_segments:
            for page, token in pdf_data.pdf_features.loop_tokens():
                if pdf_segment.intersects(PdfDataSegment.from_pdf_token(token)):
                    pdfs_tag_data.append(PdfTagData(text=token.content))

        return pdfs_tag_data

    def get_metadata_predictions(self) -> list[str]:
        start = time()
        config_logger.info("get_metadata_predictions")
        segment_selector = SegmentSelector(self.extraction_identifier)

        if not segment_selector.model or not self.pdfs_data:
            return []

        segment_selector.set_extraction_segments(self.pdfs_data)

        semantic_predictions_data = list()
        for pdf_data in self.pdfs_data:
            selected_segments = [pdf_segment for pdf_segment in pdf_data.pdf_data_segments if pdf_segment.ml_label]
            semantic_predictions_data.append(SemanticPredictionData.from_pdf_data_segments(selected_segments))

        semantic_metadata_extraction = TextToTextExtractor(self.extraction_identifier)
        semantic_predictions_texts = semantic_metadata_extraction.get_semantic_predictions(semantic_predictions_data)

        config_logger.info(f"get_semantic_predictions {round(time() - start, 2)} seconds")
        return semantic_predictions_texts if semantic_predictions_texts else []
