from pydantic import BaseModel


class Task(BaseModel):
    extraction_name: str
    tenant: str
