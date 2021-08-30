from pydantic import BaseModel


class CreateModelTask(BaseModel):
    tenant: str
    property_name: str
