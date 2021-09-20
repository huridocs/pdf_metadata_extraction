from typing import Dict

from pydantic import BaseModel


class ResultsMessage(BaseModel):
    tenant: str
    task: str
    data: Dict[str, str]
    success: bool
    error_message: str
