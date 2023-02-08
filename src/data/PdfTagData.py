from pydantic import BaseModel

from data.SegmentBox import SegmentBox


class PdfTagData(BaseModel):
    text: str
    xml_segment_box: SegmentBox = SegmentBox(left=0, top=0, width=0, height=0, page_number=1)

    @staticmethod
    def from_text(text: str):
        return PdfTagData(text=text)
