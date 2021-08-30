from pydantic import BaseModel


class Suggestion(BaseModel):
    tenant: str
    property_name: str
    xml_file_name: str
    text: str
    segment_text: str
