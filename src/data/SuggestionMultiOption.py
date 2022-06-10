from typing import List

from pydantic import BaseModel

from data.Option import Option
from data.SegmentBox import SegmentBox


class SuggestionMultiOption(BaseModel):
    tenant: str
    property_name: str
    xml_file_name: str
    options: List[Option]
    segment_text: str
    page_number: int
    segments_boxes: List[SegmentBox]
