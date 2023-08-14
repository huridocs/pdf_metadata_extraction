from pydantic import BaseModel
from pdf_token_type_labels.TokenType import TokenType
from pdf_features.Rectangle import Rectangle


class SegmentBox(BaseModel):
    left: float
    top: float
    width: float
    height: float
    page_number: int
    segment_type: TokenType = TokenType.TEXT

    def get_bounding_box(self) -> Rectangle:
        return Rectangle.from_width_height(left=int(self.left),
                                           top=int(self.top),
                                           width=int(self.width),
                                           height=int(self.height))
