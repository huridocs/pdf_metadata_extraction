from typing import List

from pydantic import BaseModel

from data.Option import Option
from data.SegmentBox import SegmentBox


class LabeledData(BaseModel):
    tenant: str
    id: str
    xml_file_name: str
    language_iso: str
    label_text: str = ""
    options: List[Option] = list()
    page_width: float
    page_height: float
    xml_segments_boxes: List[SegmentBox]
    label_segments_boxes: List[SegmentBox]

    def correct_data_scale(self):
        self.label_segments_boxes = [x.correct_input_data_scale() for x in self.label_segments_boxes]
        return self
