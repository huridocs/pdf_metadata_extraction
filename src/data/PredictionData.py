from pydantic import BaseModel

from data.SegmentBox import SegmentBox


class PredictionData(BaseModel):
    tenant: str
    id: str
    xml_file_name: str
    page_width: float
    page_height: float
    xml_segments_boxes: list[SegmentBox]

    def to_dict(self):
        prediction_data = self.model_dump()
        prediction_data["xml_segments_boxes"] = [x.to_dict() for x in self.xml_segments_boxes]
        return prediction_data
