from pydantic import BaseModel

SCALE_CONSTANT = 0.75


class SegmentBox(BaseModel):
    left: float
    top: float
    width: float
    height: float
    page_number: int

    def correct_data_scale(self):
        self.left = SCALE_CONSTANT * self.left
        self.top = SCALE_CONSTANT * self.top
        self.width = SCALE_CONSTANT * self.width
        self.height = SCALE_CONSTANT * self.height
        return self
