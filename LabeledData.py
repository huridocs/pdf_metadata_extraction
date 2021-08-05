from typing import List

from pydantic import BaseModel

from SegmentBox import SegmentBox


class LabeledData(BaseModel):
    xml_file_name: str
    extraction_name: str
    tenant: str
    label_text: str
    page_width: float
    page_height: float
    xml_segments_boxes: List[SegmentBox]
    label_segments_boxes: List[SegmentBox]
