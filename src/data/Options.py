from pydantic import BaseModel

from data.Option import Option


class Options(BaseModel):
    tenant: str
    extraction_id: str
    options: list[Option]
