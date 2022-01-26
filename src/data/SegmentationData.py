from typing import List

from pydantic import BaseModel

from data.LabeledData import LabeledData
from data.PredictionData import PredictionData
from data.SegmentBox import SegmentBox


class SegmentationData(BaseModel):
    page_width: float
    page_height: float
    xml_segments_boxes: List[SegmentBox]
    label_segments_boxes: List[SegmentBox]

    @staticmethod
    def from_labeled_data(labeled_data: LabeledData) -> "SegmentationData":
        return SegmentationData(
            page_width=labeled_data.page_width,
            page_height=labeled_data.page_height,
            xml_segments_boxes=labeled_data.xml_segments_boxes,
            label_segments_boxes=labeled_data.label_segments_boxes,
        )

    @staticmethod
    def from_prediction_data(prediction_data: PredictionData) -> "SegmentationData":
        return SegmentationData(
            page_width=prediction_data.page_width,
            page_height=prediction_data.page_height,
            xml_segments_boxes=prediction_data.xml_segments_boxes,
            label_segments_boxes=[],
        )
