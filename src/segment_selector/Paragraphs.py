from pydantic import BaseModel

from segment_selector.ParagraphSegmentBox import ParagraphSegmentBox


class Paragraphs(BaseModel):
    page_width: int
    page_height: int
    paragraphs: list[ParagraphSegmentBox]
