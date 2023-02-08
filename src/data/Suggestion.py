from typing import List

from pydantic import BaseModel

from data.Option import Option
from data.SegmentBox import SegmentBox
from data.SemanticPredictionData import SemanticPredictionData


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
        tenant: str, property_name: str, semantic_prediction_data: SemanticPredictionData, prediction: str
    ):

        page_number = 1

        if semantic_prediction_data.pdf_tags and semantic_prediction_data.pdf_tags[0].xml_segment_box:
            page_number = semantic_prediction_data.pdf_tags[0].xml_segment_box.page_number

        return Suggestion(
            tenant=tenant,
            property_name=property_name,
            xml_file_name=semantic_prediction_data.xml_file_name,
            text=prediction,
            segment_text=" ".join([x.text for x in semantic_prediction_data.pdf_tags]),
            page_number=page_number,
            segments_boxes=[x.xml_segment_box.correct_output_data_scale() for x in semantic_prediction_data.pdf_tags],
        )
