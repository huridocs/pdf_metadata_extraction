from pydantic import BaseModel

SCALE_CONSTANT = 0.75


class SegmentBox(BaseModel):
    left: float
    top: float
    width: float
    height: float
    page_number: int

    def correct_input_data_scale(self):
        self.left = self.left * SCALE_CONSTANT
        self.top = self.top * SCALE_CONSTANT
        self.width = self.width * SCALE_CONSTANT
        self.height = self.height * SCALE_CONSTANT
        return self

    def correct_output_data_scale(self):
        self.left = self.left / SCALE_CONSTANT
        self.top = self.top / SCALE_CONSTANT
        self.width = self.width / SCALE_CONSTANT
        self.height = self.height / SCALE_CONSTANT
        return self
