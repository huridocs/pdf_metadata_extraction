import os

from fastapi.testclient import TestClient
from unittest import TestCase
from app import app, LabeledDataParams

client = TestClient(app)


class TestApp(TestCase):
    def test_info(self):
        response = client.get("/info")
        self.assertEqual(200, response.status_code)

    def test_error(self):
        response = client.get("/error")
        self.assertEqual(500, response.status_code)
        self.assertEqual({'detail': 'This is a test error from the error endpoint'}, response.json())

    def test_label(self):
        with open('test_pdf/test.pdf', 'rb') as stream:
            files = {'file': stream}
            response = client.post("/", json={"label_id": "label_id"})
            self.assertEqual('test.pdflabel_id', response.json())

    def test_labeled_xml(self):
        with open('test_pdf/test.pdf', 'rb') as stream:
            files = {'file': stream}
            response = client.post("/labeled_xml/tenant_one", files=files)
            self.assertEqual('task registered', response.json())
            self.assertEqual(200, response.status_code)
            self.assertTrue(os.path.exists('./docker_volume/labeled_xmls/tenant_one/test.pdf'))
            os.remove('./docker_volume/labeled_xmls/tenant_one/test.pdf')
            os.rmdir('./docker_volume/labeled_xmls/tenant_one')