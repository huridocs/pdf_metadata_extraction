import os

import mongomock
import pymongo
from fastapi.testclient import TestClient
from unittest import TestCase
from app import app

client = TestClient(app)


class TestApp(TestCase):
    def test_info(self):
        response = client.get("/info")
        self.assertEqual(200, response.status_code)

    def test_error(self):
        response = client.get("/error")
        self.assertEqual(500, response.status_code)
        self.assertEqual({'detail': 'This is a test error from the error endpoint'}, response.json())

    def test_xml_file(self):
        with open('test_pdf/test.pdf', 'rb') as stream:
            files = {'file': stream}
            response = client.post("/xml_file/tenant_one", files=files)
            self.assertEqual('task registered', response.json())
            self.assertEqual(200, response.status_code)
            self.assertTrue(os.path.exists('./docker_volume/tenant_one/xml_files/test.pdf'))
            os.remove('./docker_volume/tenant_one/xml_files/test.pdf')
            os.rmdir('./docker_volume/tenant_one/xml_files')
            os.rmdir('./docker_volume/tenant_one')

    @mongomock.patch(servers=['mongodb://mongo:27017'])
    def test_labeled_data(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo:27017')

        json_data = {"xml_file_name": "xml_file_name",
                     "extraction_name": "extraction_name",
                     "tenant": "tenant",
                     "label_text": "text",
                     "page_width": 1.1,
                     "page_height": 2.1,
                     "xml_segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 5}],
                     "label_segments_boxes": [{"left": 6, "top": 7, "width": 8, "height": 9, "page_number": 10}]
                     }

        response = client.post("/labeled_data", json=json_data)

        labeled_data_document = mongo_client.pdf_information_extraction.labeleddata.find_one()

        self.assertEqual(200, response.status_code)
        self.assertEqual('extraction_name', labeled_data_document['extraction_name'])
        self.assertEqual('tenant', labeled_data_document['tenant'])
        self.assertEqual('text', labeled_data_document['label_text'])
        self.assertEqual(1.1, labeled_data_document['page_width'])
        self.assertEqual(2.1, labeled_data_document['page_height'])
        self.assertEqual('xml_file_name', labeled_data_document['xml_file_name'])
        self.assertEqual('text', labeled_data_document['label_text'])
        self.assertEqual([{'height': 4.0, 'left': 1.0, 'page_number': 5, 'top': 2.0, 'width': 3.0}],
                         labeled_data_document['xml_segments_boxes'])
        self.assertEqual([{'height': 9.0, 'left': 6.0, 'page_number': 10, 'top': 7.0, 'width': 8.0}],
                         labeled_data_document['label_segments_boxes'])

    def test_error_labeled_data(self):
        json_data = {"xml_file_name": "xml_file_name",
                     "extraction_name": "extraction_name",
                     "tenant": "tenant",
                     "label_text": "text",
                     "page_width": 1.1,
                     "page_height": 2.1,
                     "xml_segments_boxes": [{"left": "string_value", "top": 2, "width": 3, "height": 4, "page_number": 5}],
                     "label_segments_boxes": [{"left": 6, "top": 7, "width": 8, "height": 9, "page_number": 10}]
                     }

        response = client.post("/labeled_data", json=json_data)

        print(response.json())
        self.assertEqual(422, response.status_code)

    @mongomock.patch(servers=['mongodb://mongo:27017'])
    def test_labeled_data_different_values(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo:27017')

        json_data = {"xml_file_name": "other_xml_file_name",
                     "extraction_name": "other_extraction_name",
                     "tenant": "other_tenant",
                     "label_text": "other_text",
                     "page_width": 4.1,
                     "page_height": 5.1,
                     "xml_segments_boxes": [],
                     "label_segments_boxes": []
                     }

        response = client.post("/labeled_data", json=json_data)

        labeled_data_document = mongo_client.pdf_information_extraction.labeleddata.find_one()

        self.assertEqual(200, response.status_code)
        self.assertEqual('other_extraction_name', labeled_data_document['extraction_name'])
        self.assertEqual('other_tenant', labeled_data_document['tenant'])
        self.assertEqual('other_text', labeled_data_document['label_text'])
        self.assertEqual(4.1, labeled_data_document['page_width'])
        self.assertEqual(5.1, labeled_data_document['page_height'])
        self.assertEqual('other_xml_file_name', labeled_data_document['xml_file_name'])
        self.assertEqual('other_text', labeled_data_document['label_text'])
        self.assertEqual([], labeled_data_document['xml_segments_boxes'])
        self.assertEqual([], labeled_data_document['label_segments_boxes'])

    @mongomock.patch(servers=['mongodb://mongo:27017'])
    def test_prediction_data(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo:27017')

        json_data = {"xml_file_name": "xml_file_name",
                     "extraction_name": "extraction_name",
                     "tenant": "tenant",
                     }

        response = client.post("/prediction_data", json=json_data)

        prediction_data_document = mongo_client.pdf_information_extraction.predictiondata.find_one()

        self.assertEqual(200, response.status_code)
        self.assertEqual('extraction_name', prediction_data_document['extraction_name'])
        self.assertEqual('tenant', prediction_data_document['tenant'])
        self.assertEqual('xml_file_name', prediction_data_document['xml_file_name'])