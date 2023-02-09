from typing import List

from pydantic import BaseModel

from data.Option import Option
from data.SegmentBox import SegmentBox
from data.SemanticPredictionData import SemanticPredictionData
from metadata_extraction.PdfFeatures.PdfFeatures import PdfFeatures
from metadata_extraction.PdfFeatures.PdfSegment import PdfSegment


class Suggestion(BaseModel):
    tenant: str
    property_name: str
    xml_file_name: str
    text: str = ""
    options: List[Option] = list()
    segment_text: str
    page_number: int
    segments_boxes: List[SegmentBox]

    @staticmethod
    def from_prediction_data(
        tenant: str,
        property_name: str,
        semantic_prediction_data: SemanticPredictionData,
        prediction: str,
        pdf_features: PdfFeatures,
    ):
        segments = [x for x in pdf_features.pdf_segments if x.ml_label]

        if segments:
            page_number = segments[0].page_number
        else:
            page_number = 1

        return Suggestion(
            tenant=tenant,
            property_name=property_name,
            xml_file_name=semantic_prediction_data.xml_file_name,
            text=prediction,
            segment_text=" ".join([x.text for x in semantic_prediction_data.pdf_tags]),
            page_number=page_number,
            segments_boxes=[x.bounding_box.to_segment_box(x.page_number).correct_output_data_scale() for x in segments],
        )

    @staticmethod
    def get_empty(tenant: str, property_name: str, xml_file_name: str) -> "Suggestion":
        return Suggestion(
            tenant=tenant,
            property_name=property_name,
            xml_file_name=xml_file_name,
            text="",
            segment_text=" ",
            page_number=1,
            segments_boxes=list(),
        )

    def add_prediction(self, text: str):
        self.text = text
        return self

    def add_segments(self, pdf_features):
        segments: list[PdfSegment] = [x for x in pdf_features.pdf_segments if x.ml_label]

        if segments:
            self.page_number = segments[0].page_number
        else:
            self.page_number = 1

        self.segments_boxes = [x.bounding_box.to_segment_box(x.page_number).correct_output_data_scale() for x in segments]
        self.segment_text = " ".join([x.text_content for x in segments])
        return self
