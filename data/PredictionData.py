from typing import List

from pydantic import BaseModel

from data.SegmentBox import SegmentBox


class PredictionData(BaseModel):
    tenant: str
    template: str
    property_name: str
    xml_file_name: str
    page_width: float
    page_height: float
    xml_segments_boxes: List[SegmentBox]
