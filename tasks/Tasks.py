import os
import pathlib
from pathlib import Path

import pymongo

from data.CreateModelTask import CreateModelTask


class Tasks:
    def __init__(self, tenant: str, template: str):
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.root_folder = path.parent.absolute()
        self.tenant = tenant
        self.template = template
        client = pymongo.MongoClient('mongodb://mongo_paragraphs:27017')
        self.prediction_tasks_collection = client['pdf_paragraph'].prediction_tasks

    def add_prediction_task(self, pdf_file_name, file):
        if not os.path.exists(f'{self.root_folder}/docker_volume/to_extract'):
            os.mkdir(f'{self.root_folder}/docker_volume/to_extract')

        if not os.path.exists(f'{self.root_folder}/docker_volume/to_extract/{self.tenant}'):
            os.mkdir(f'{self.root_folder}/docker_volume/to_extract/{self.tenant}')

        path = f'{self.root_folder}/docker_volume/to_extract/{self.tenant}/{pdf_file_name}'

        file_path_pdf = pathlib.Path(path)
        file_path_pdf.write_bytes(file)
        task = CreateModelTask(tenant=self.tenant, pdf_file_name=pdf_file_name)

        self.prediction_tasks_collection.delete_many({'tenant': task.tenant, 'pdf_file_name': pdf_file_name})
        self.prediction_tasks_collection.insert_one(task.dict())

    @classmethod
    def get_next_task(cls) -> CreateModelTask:
        client = pymongo.MongoClient('mongodb://mongo_paragraphs:27017')
        document = client['pdf_paragraph'].tasks.find_one_and_delete({})

        if not document:
            return None

        return CreateModelTask(**document)
