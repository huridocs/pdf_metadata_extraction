from pydantic import BaseModel


class CreateModelTask(BaseModel):
    tenant: str
    template: str
    property_name: str
