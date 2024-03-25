import os
from os.path import join, exists
from time import time

from pydantic import BaseModel

from config import DATA_PATH


class ExtractionIdentifier(BaseModel):
    run_name: str
    extraction_name: str

    def get_path(self):
        return join(DATA_PATH, self.run_name, self.extraction_name)

    def is_old(self):
        path = self.get_path()
        return exists(path) and os.path.isdir(path) and os.path.getmtime(path) < (time() - (2 * 24 * 3600))
