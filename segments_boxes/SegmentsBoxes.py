import json
from typing import List


class SegmentBox:
    def __init__(self, page_number, left, top, width, height, ):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.page_number = page_number

    def to_dictionary(self):
        return {"pageNumber": self.page_number,
                "left": self.left, "top": self.top, "width": self.width, "height": self.height}


class SegmentsBoxes:
    def __init__(self, page_width, page_height, segment_boxes: List[SegmentBox], ):
        self.page_width = page_width
        self.page_height = page_height
        self.segment_boxes = segment_boxes

    def to_json(self):
        segments = [segment_box.to_dictionary() for segment_box in self.segment_boxes]
        segments_boxes_json = {"pageWidth": self.page_width, "pageHeight": self.page_height, "segments": segments}
        return json.dumps(segments_boxes_json)

