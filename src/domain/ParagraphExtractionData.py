from pydantic import BaseModel
from trainable_entity_extractor.data.SegmentBox import SegmentBox

from domain.XML import XML


class XmlSegments(XML):
    xml_segments_boxes: list[SegmentBox]


class ParagraphExtractionData(BaseModel):
    key: str
    xmls_segments: list[XmlSegments]
