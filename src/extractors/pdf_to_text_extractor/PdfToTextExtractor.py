from time import time

from config import config_logger
from data.ExtractionData import ExtractionData
from data.PdfData import PdfData
from data.PdfDataSegment import PdfDataSegment
from data.PredictionSample import PredictionSample
from data.Suggestion import Suggestion
from extractors.ExtractorBase import ExtractorBase
from extractors.segment_selector.SegmentSelector import SegmentSelector
from extractors.text_to_text_extractor.TextToTextExtractor import TextToTextExtractor


class PdfToTextExtractor(ExtractorBase):

    def is_valid(self, extraction_data: ExtractionData) -> bool:
        for sample in extraction_data.samples:
            if sample.pdf_data:
                return True

        return False

    def create_model(self, extraction_data: ExtractionData) -> tuple[bool, str]:
        all_pdf_segments: list[PdfDataSegment] = [
            pdf_segment for sample in extraction_data.samples for pdf_segment in sample.pdf_data.pdf_data_segments
        ]

        if not all_pdf_segments:
            return False, "No data to create model"

        success, error = self.create_segment_selector_model(extraction_data)

        if not success:
            return False, error

        success, error = self.create_semantic_model(extraction_data)
        return success, error

    def create_segment_selector_model(self, extraction_data):
        segment_selector = SegmentSelector(self.extraction_identifier)
        pdfs_data = [sample.pdf_data for sample in extraction_data.samples]
        return segment_selector.create_model(pdfs_data=pdfs_data)

    def create_semantic_model(self, extraction_data: ExtractionData):
        semantic_metadata_extraction = TextToTextExtractor(self.extraction_identifier)
        semantic_metadata_extraction.remove_models()
        for sample in extraction_data.samples:
            sample.tags_texts = self.get_predicted_texts(sample.pdf_data)

        return semantic_metadata_extraction.create_model(extraction_data)

    @staticmethod
    def get_predicted_texts(pdf_data: PdfData) -> list[str]:
        predicted_pdf_segments = [x for x in pdf_data.pdf_data_segments if x.ml_label]

        tags_texts: list[str] = list()
        for pdf_segment in predicted_pdf_segments:
            for page, token in pdf_data.pdf_features.loop_tokens():
                if pdf_segment.intersects(PdfDataSegment.from_pdf_token(token)):
                    tags_texts.append(token.content.strip())

        return tags_texts

    def get_suggestions(self, predictions_samples: list[PredictionSample]) -> list[Suggestion]:
        start = time()
        config_logger.info("get_metadata_predictions")
        segment_selector = SegmentSelector(self.extraction_identifier)

        if not segment_selector.model or not predictions_samples:
            return []

        segment_selector.set_extraction_segments([sample.pdf_data for sample in predictions_samples])

        for sample in predictions_samples:
            sample.tags_texts = self.get_predicted_texts(sample.pdf_data)

        semantic_metadata_extraction = TextToTextExtractor(self.extraction_identifier)
        suggestions = semantic_metadata_extraction.get_suggestions(predictions_samples)

        for suggestion, sample in zip(suggestions, predictions_samples):
            suggestion.add_segments(sample.pdf_data)

        config_logger.info(f"get_semantic_predictions {round(time() - start, 2)} seconds")
        return suggestions
