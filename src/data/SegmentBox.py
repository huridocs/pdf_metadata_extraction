from pydantic import BaseModel

SCALE_CONSTANT = 0.75


class SegmentBox(BaseModel):
    left: float
    top: float
    width: float
    height: float
    page_number: int

    def correct_input_data_scale(self):
        return self.rescaled(SCALE_CONSTANT, SCALE_CONSTANT)

    def correct_output_data_scale(self):
        return self.rescaled(1/SCALE_CONSTANT, 1/SCALE_CONSTANT)

    def rescaled(self, factor_width: float, factor_height: float):
        self.left = self.left * factor_width
        self.top = self.top * factor_height
        self.width = self.width * factor_width
        self.height = self.height * factor_height
        return self
