import os
import pathlib
from pathlib import Path


class XmlFile:
    def __init__(self, filename: str, tenant: str):
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.root_folder = path.parent.absolute()
        self.tenant = tenant
        self.filename = filename

    def save(self, file: bytes):
        if not os.path.exists(f'{self.root_folder}/docker_volume/{self.tenant}'):
            os.mkdir(f'{self.root_folder}/docker_volume/{self.tenant}')

        if not os.path.exists(f'{self.root_folder}/docker_volume/{self.tenant}/xml_files'):
            os.mkdir(f'{self.root_folder}/docker_volume/{self.tenant}/xml_files')

        path = f'{self.root_folder}/docker_volume/{self.tenant}/xml_files/{self.filename}'

        file_path_pdf = pathlib.Path(path)
        file_path_pdf.write_bytes(file)
