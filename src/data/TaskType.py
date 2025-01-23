from pydantic import BaseModel


class TaskType(BaseModel):
    task: str
