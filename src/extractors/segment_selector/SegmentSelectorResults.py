from pydantic import BaseModel


class SegmentSelectorResults(BaseModel):
    method: str
    dataset: str
    precision: float
    recall: float
    seconds: int

    @staticmethod
    def get_padding():
        return {"method": "right", "dataset": "right", "precision": "left", "recall": "left", "seconds": "left"}
