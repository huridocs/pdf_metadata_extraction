from pydantic import BaseModel
from trainable_entity_extractor.domain.SegmentBox import SegmentBox


class XmlData(BaseModel):
    xml_file_name: str
    language: str
    is_main_language: bool
    xml_segments_boxes: list[SegmentBox]


class ParagraphExtractionData(BaseModel):
    key: str
    xmls: list[XmlData]
