from pydantic import BaseModel


class Suggestion(BaseModel):
    extraction_name: str
    xml_file_name: str
    tenant: str
    text: str
    segment_text: str
