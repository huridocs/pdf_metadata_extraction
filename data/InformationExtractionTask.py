from typing import Dict

from pydantic import BaseModel


class InformationExtractionTask(BaseModel):
    tenant: str
    task: str
    data: Dict[str, str]