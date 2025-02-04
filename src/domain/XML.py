from pydantic import BaseModel


class XML(BaseModel):
    xml_file_name: str
    language: str
    is_main_language: bool
