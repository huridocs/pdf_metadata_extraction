from typing import List

from pydantic import BaseModel

from SegmentBox import SegmentBox


class LabeledData(BaseModel):
    extraction_name: str
    xml_file_name: str
    tenant: str
    label_text: str
    page_width: float
    page_height: float
    xml_segments_boxes: List[SegmentBox]
    label_segments_boxes: List[SegmentBox]
