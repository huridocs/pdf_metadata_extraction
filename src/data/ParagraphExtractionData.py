from pydantic import BaseModel
from trainable_entity_extractor.data.SegmentBox import SegmentBox

from config import PARAGRAPH_EXTRACTION_NAME
from data.XML import XML


class XmlData(XML):
    xml_segments_boxes: list[SegmentBox]


class ParagraphExtractionData(BaseModel):
    key: str
    xmls: list[XmlData]

    def to_db(self):
        db_model = self.model_dump()
        db_model["run_name"] = PARAGRAPH_EXTRACTION_NAME
        db_model["extraction_name"] = self.key
        return db_model
