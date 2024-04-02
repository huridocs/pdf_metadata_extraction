from pdf_token_type_labels.TokenType import TokenType
from pydantic import BaseModel

from data.ExtractionIdentifier import ExtractionIdentifier
from data.Option import Option
from data.SegmentBox import SegmentBox
from data.SemanticPredictionData import SemanticPredictionData
from data.PdfDataSegment import PdfDataSegment
from data.PdfData import PdfData
from data.ExtractionSample import ExtractionSample
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.Beginning750 import Beginning750


class Suggestion(BaseModel):
    tenant: str
    id: str
    xml_file_name: str = ""
    text: str = ""
    values: list[Option] = list()
    segment_text: str
    page_number: int
    segments_boxes: list[SegmentBox]

    def to_dict(self):
        suggestion_dict = self.model_dump()
        suggestion_dict["segments_boxes"] = [x.to_dict() for x in self.segments_boxes]
        return suggestion_dict

    @staticmethod
    def from_prediction_data(
        tenant: str,
        extraction_id: str,
        semantic_prediction_data: SemanticPredictionData,
        prediction: str,
        pdf_segments: PdfData,
    ):
        segments = [x for x in pdf_segments.pdf_data_segments if x.ml_label]

        if segments:
            page_number = segments[0].page_number
        else:
            page_number = 1

        return Suggestion(
            tenant=tenant,
            id=extraction_id,
            xml_file_name=semantic_prediction_data.xml_file_name,
            text=prediction,
            segment_text=" ".join([x.text for x in semantic_prediction_data.pdf_tags_data]),
            page_number=page_number,
            segments_boxes=[x.bounding_box.to_segment_box(x.page_number).correct_output_data_scale() for x in segments],
        )

    @staticmethod
    def get_empty(extraction_identifier: ExtractionIdentifier, xml_file_name: str) -> "Suggestion":
        return Suggestion(
            tenant=extraction_identifier.run_name,
            id=extraction_identifier.extraction_name,
            xml_file_name=xml_file_name,
            text="",
            segment_text=" ",
            page_number=1,
            segments_boxes=list(),
        )

    def add_prediction(self, text: str, prediction_pdf_data: PdfData):
        self.add_segments(prediction_pdf_data)
        self.text = text

    def add_prediction_multi_option(self, prediction_sample: ExtractionSample, values: list[Option]):
        self.add_segments(prediction_sample.pdf_data)
        self.values = values

    def add_segments(self, pdf_data: PdfData):
        context_segments: list[PdfDataSegment] = [x for x in pdf_data.pdf_data_segments if x.ml_label]
        valid_types = [TokenType.TEXT, TokenType.TITLE, TokenType.HEADER]
        context_segments = [x for x in context_segments if x.segment_type in valid_types]

        if not context_segments:
            self.page_number = 1
            return

        context_segments = Beginning750().filter_segments(context_segments)
        self.page_number = context_segments[0].page_number
        self.segments_boxes = [SegmentBox.from_pdf_segment(pdf_segment) for pdf_segment in context_segments]
        self.segment_text = " ... ".join([pdf_segment.text_content for pdf_segment in context_segments])

    def scale_up(self):
        for segment_box in self.segments_boxes:
            segment_box.scale_up()

        return self
