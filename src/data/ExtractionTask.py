from pydantic import BaseModel
from data.Params import Params


class ExtractionTask(BaseModel):
    tenant: str
    task: str
    params: Params
