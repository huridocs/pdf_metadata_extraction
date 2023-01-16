import os
import shutil
from os.path import join
from unittest import TestCase

from config import APP_PATH
from metadata_extraction.XmlFile import XmlFile

DOCKER_VOLUME_PATH = (
    f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))}/docker_volume"
)


class TestXmlFile(TestCase):
    test_file_path = f"{APP_PATH}/tenant_test/property_name/xml_to_train/test.xml"

    def test_save_xml_to_train(self):
        tenant = "tenant_save"
        property_name = "property_save"

        base_path = f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)
        with open(self.test_file_path, "rb") as file:
            xml_file = XmlFile(
                tenant=tenant,
                property_name=property_name,
                to_train=True,
                xml_file_name="test.xml",
            )

            xml_file.save(file=file.read())

        self.assertTrue(os.path.exists(f"{base_path}/xml_to_train/test.xml"))

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

    def test_save_xml_to_predict(self):
        tenant = "tenant_save"
        property_name = "property_save"

        base_path = f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)
        with open(self.test_file_path, "rb") as file:
            xml_file = XmlFile(
                tenant=tenant,
                property_name=property_name,
                to_train=False,
                xml_file_name="test.xml",
            )

            xml_file.save(file=file.read())

        self.assertTrue(os.path.exists(f"{base_path}/xml_to_predict/test.xml"))

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)
