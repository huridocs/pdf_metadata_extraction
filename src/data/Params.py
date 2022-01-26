from pydantic import BaseModel


class Params(BaseModel):
    property_name: str
