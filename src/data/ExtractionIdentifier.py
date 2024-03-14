from os.path import join

from pydantic import BaseModel

from config import DATA_PATH


class ExtractionIdentifier(BaseModel):
    run_name: str
    extraction_name: str

    def get_path(self):
        return join(DATA_PATH, self.run_name, self.extraction_name)
