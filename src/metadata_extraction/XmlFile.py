import os
import pathlib

from ServiceConfig import ServiceConfig


class XmlFile:
    def __init__(self, tenant: str, property_name: str, to_train: bool, xml_file_name: str):
        self.tenant = tenant
        self.property_name = property_name
        self.to_train = to_train
        self.xml_file_name = xml_file_name
        self.xml_file = None
        self.xml_folder_path = XmlFile.get_xml_folder_path(tenant, property_name, to_train)
        self.xml_file_path = os.path.join(self.xml_folder_path, self.xml_file_name)

    def save(self, file: bytes):
        if not os.path.exists(self.xml_folder_path):
            os.makedirs(self.xml_folder_path)

        file_path = pathlib.Path(f"{self.xml_folder_path}/{self.xml_file_name}")
        file_path.write_bytes(file)

    @staticmethod
    def get_xml_folder_path(tenant: str, property_name: str, to_train: bool) -> str:
        service_config = ServiceConfig()
        xml_folder_path = f"{service_config.docker_volume_path}/{tenant}/{property_name}"
        if to_train:
            xml_folder_path += f"/xml_to_train"
        else:
            xml_folder_path += f"/xml_to_predict"

        return xml_folder_path
