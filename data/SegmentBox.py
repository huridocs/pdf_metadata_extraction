from pydantic import BaseModel


class SegmentBox(BaseModel):
    left: float
    top: float
    width: float
    height: float
    page_number: int
