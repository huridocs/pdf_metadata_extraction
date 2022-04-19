from typing import List

from pydantic import BaseModel

from data.SegmentBox import SegmentBox


class Suggestion(BaseModel):
    tenant: str
    property_name: str
    xml_file_name: str
    text: str
    segment_text: str
    page_number: int
    segments_boxes: List[SegmentBox]
