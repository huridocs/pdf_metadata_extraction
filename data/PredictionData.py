from pydantic import BaseModel


class PredictionData(BaseModel):
    extraction_name: str
    xml_file_name: str
    tenant: str
