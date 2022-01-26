from pydantic import BaseModel
from data.Params import Params


class InformationExtractionTask(BaseModel):
    tenant: str
    task: str
    params: Params
