import json

from pydantic import BaseModel
from pdf_token_type_labels.TokenType import TokenType
from pdf_features.Rectangle import Rectangle

SCALE_RATIO = 0.75


class SegmentBox(BaseModel):
    left: float
    top: float
    width: float
    height: float
    page_number: int
    segment_type: TokenType = TokenType.TEXT

    def to_dict(self):
        return json.loads(self.model_dump_json())

    def get_bounding_box(self) -> Rectangle:
        return Rectangle.from_width_height(
            left=int(self.left), top=int(self.top), width=int(self.width), height=int(self.height)
        )

    def scale_down(self):
        self.left = round(self.left * SCALE_RATIO, 0)
        self.top = round(self.top * SCALE_RATIO, 0)
        self.width = round(self.width * SCALE_RATIO, 0)
        self.height = round(self.height * SCALE_RATIO, 0)

    def scale_up(self):
        self.left = round(self.left / SCALE_RATIO, 0)
        self.top = round(self.top / SCALE_RATIO, 0)
        self.width = round(self.width / SCALE_RATIO, 0)
        self.height = round(self.height / SCALE_RATIO, 0)
