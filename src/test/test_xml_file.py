import os
import shutil
from os.path import join
from unittest import TestCase

from config import APP_PATH, DATA_PATH
from data.ExtractionIdentifier import ExtractionIdentifier
from XmlFile import XmlFile


class TestXmlFile(TestCase):
    test_file_path = f"{APP_PATH}/tenant_test/extraction_id/xml_to_train/test.xml"

    def test_save_xml_to_train(self):
        tenant = "tenant_save"
        extraction_id = "property_save"

        base_path = f"{DATA_PATH}/{tenant}/{extraction_id}"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
        with open(self.test_file_path, "rb") as file:
            xml_file = XmlFile(
                extraction_identifier=ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id),
                to_train=True,
                xml_file_name="test.xml",
            )

            xml_file.save(file=file.read())

        self.assertTrue(os.path.exists(f"{base_path}/xml_to_train/test.xml"))

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

    def test_save_xml_to_predict(self):
        tenant = "tenant_save"
        extraction_id = "property_save"

        base_path = f"{DATA_PATH}/{tenant}/{extraction_id}"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
        with open(self.test_file_path, "rb") as file:
            xml_file = XmlFile(
                extraction_identifier=ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id),
                to_train=False,
                xml_file_name="test.xml",
            )

            xml_file.save(file=file.read())

        self.assertTrue(os.path.exists(f"{base_path}/xml_to_predict/test.xml"))

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
