from pydantic import BaseModel

from domain.XML import XML


class ParagraphExtractionResultsMessage(BaseModel):
    key: str
    xmls: list[XML]
    success: bool
    error_message: str
    data_url: str = None
