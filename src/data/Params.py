from typing import List

from pydantic import BaseModel

from data.Option import Option


class Params(BaseModel):
    property_name: str
    options: List[Option] = list()
    multi_value: bool = False
