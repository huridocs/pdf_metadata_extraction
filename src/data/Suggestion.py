from pdf_token_type_labels.TokenType import TokenType
from pydantic import BaseModel

from data.ExtractionIdentifier import ExtractionIdentifier
from data.Option import Option
from data.SegmentBox import SegmentBox
from data.PdfDataSegment import PdfDataSegment
from data.PdfData import PdfData
from data.TrainingSample import TrainingSample
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.Beginning750 import Beginning750
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.End750 import End750


class Suggestion(BaseModel):
    tenant: str
    id: str
    xml_file_name: str = ""
    entity_name: str = ""
    text: str = ""
    values: list[Option] = list()
    segment_text: str = ""
    page_number: int = 1
    segments_boxes: list[SegmentBox] = list()

    def to_dict(self):
        suggestion_dict = self.model_dump()
        suggestion_dict["segments_boxes"] = [x.to_dict() for x in self.segments_boxes]
        return suggestion_dict

    @staticmethod
    def get_empty(extraction_identifier: ExtractionIdentifier, entity_name: str) -> "Suggestion":
        return Suggestion(
            tenant=extraction_identifier.run_name,
            id=extraction_identifier.extraction_name,
            xml_file_name=entity_name,
            entity_name=entity_name,
        )

    def add_prediction(self, text: str, prediction_pdf_data: PdfData):
        self.add_segments(prediction_pdf_data)
        self.text = text

    def add_prediction_multi_option(self, training_sample: TrainingSample, values: list[Option], context_from_the_end: bool):
        self.add_segments(training_sample.pdf_data, context_from_the_end)
        self.values = values

    def add_segments(self, pdf_data: PdfData, context_from_the_end: bool = False):
        context_segments: list[PdfDataSegment] = [x for x in pdf_data.pdf_data_segments if x.ml_label]
        valid_types = [TokenType.LIST_ITEM, TokenType.TITLE, TokenType.TEXT, TokenType.SECTION_HEADER, TokenType.CAPTION]
        context_segments = [x for x in context_segments if x.segment_type in valid_types]

        if not context_segments:
            self.page_number = 1
            return

        if context_from_the_end:
            context_segments = End750().filter_segments(context_segments)
        else:
            context_segments = Beginning750().filter_segments(context_segments)

        self.page_number = context_segments[0].page_number
        self.segments_boxes = [SegmentBox.from_pdf_segment(pdf_segment) for pdf_segment in context_segments]
        self.segment_text = " .. ".join([pdf_segment.text_content for pdf_segment in context_segments])

    def scale_up(self):
        for segment_box in self.segments_boxes:
            segment_box.scale_up()

        return self

    @staticmethod
    def from_prediction_text(extraction_identifier: ExtractionIdentifier, entity_name: str, text: str):
        suggestion = Suggestion.get_empty(extraction_identifier, entity_name)
        suggestion.text = text
        return suggestion

    @staticmethod
    def from_prediction_multi_option(extraction_identifier: ExtractionIdentifier, entity_name: str, values: list[Option]):
        suggestion = Suggestion.get_empty(extraction_identifier, entity_name)
        suggestion.values = values
        return suggestion
