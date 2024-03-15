from pydantic import BaseModel

from data.ExtractionIdentifier import ExtractionIdentifier
from data.Option import Option
from data.SegmentBox import SegmentBox
from data.SemanticPredictionData import SemanticPredictionData
from metadata_extraction.PdfDataSegment import PdfDataSegment
from metadata_extraction.PdfData import PdfData


class Suggestion(BaseModel):
    tenant: str
    id: str
    xml_file_name: str
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

    def add_prediction(self, text: str):
        self.text = text
        return self

    def add_prediction_multi_option(self, options: list[Option]):
        self.values = options
        return self

    def add_segments(self, pdf_data: PdfData):
        segments: list[PdfDataSegment] = [x for x in pdf_data.pdf_data_segments if x.ml_label]

        if segments:
            self.page_number = segments[0].page_number
        else:
            self.page_number = 1

        self.segments_boxes = [SegmentBox.from_pdf_segment(pdf_segment) for pdf_segment in segments]
        self.segment_text = " ".join([pdf_segment.text_content for pdf_segment in segments])
        return self

    def scale_up(self):
        for segment_box in self.segments_boxes:
            segment_box.scale_up()

        return self
