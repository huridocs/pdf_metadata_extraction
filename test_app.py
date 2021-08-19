import json
import os
import shutil
from typing import List

import mongomock
import pymongo
from fastapi.testclient import TestClient
from unittest import TestCase
from app import app
from data.SemanticExtractionData import SemanticExtractionData
from data.Suggestion import Suggestion
from semantic_information_extraction.SemanticInformationExtraction import SemanticInformationExtraction

client = TestClient(app)

DOCKER_VOLUME_PATH = f'./docker_volume'


class TestApp(TestCase):
    def test_info(self):
        response = client.get("/info")
        self.assertEqual(200, response.status_code)

    def test_error(self):
        response = client.get("/error")
        self.assertEqual(500, response.status_code)
        self.assertEqual({'detail': 'This is a test error from the error endpoint'}, response.json())

    def test_post_xml_file(self):
        with open('DOCKER_VOLUME_PATH/tenant_test/extraction_name/xml_files/test.xml', 'rb') as stream:
            files = {'file': stream}
            response = client.post("/xml_file/tenant%20one/extraction%20name", files=files)

        self.assertEqual('xml saved', response.json())
        self.assertEqual(200, response.status_code)
        self.assertTrue(os.path.exists(f'{DOCKER_VOLUME_PATH}/tenant_one/extraction_name/xml_files/test.xml'))

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/tenant_one', ignore_errors=True)

    @mongomock.patch(servers=['mongodb://mongo_information_extraction:27017'])
    def test_post_labeled_data(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')

        json_data = {"xml_file_name": "xml_file_name",
                     "extraction_name": "extraction name",
                     "tenant": "tenant one",
                     "language_iso" : "en",
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
        self.assertEqual('tenant_one', labeled_data_document['tenant'])
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
                     "xml_segments_boxes": [
                         {"left": "string_value", "top": 2, "width": 3, "height": 4, "page_number": 5}],
                     "label_segments_boxes": [{"left": 6, "top": 7, "width": 8, "height": 9, "page_number": 10}]
                     }

        response = client.post("/labeled_data", json=json_data)

        self.assertEqual(422, response.status_code)

    @mongomock.patch(servers=['mongodb://mongo_information_extraction:27017'])
    def test_post_labeled_data_different_values(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')

        json_data = {"xml_file_name": "other_xml_file_name",
                     "extraction_name": "other_extraction_name",
                     "tenant": "other tenant",
                     "language_iso": "en",
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

    @mongomock.patch(servers=['mongodb://mongo_information_extraction:27017'])
    def test_post_prediction_data(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')

        json_data = {"xml_file_name": "xml_file_name",
                     "extraction_name": "extraction name",
                     "tenant": "tenant",
                     "page_width": 612,
                     "page_height": 792,
                     "xml_segments_boxes": [{"left": 6, "top": 7, "width": 8, "height": 9, "page_number": 10}],
                     }

        response = client.post("/prediction_data", json=json_data)

        prediction_data_document = mongo_client.pdf_information_extraction.predictiondata.find_one()

        self.assertEqual(200, response.status_code)
        self.assertEqual('extraction_name', prediction_data_document['extraction_name'])
        self.assertEqual('tenant', prediction_data_document['tenant'])
        self.assertEqual('xml_file_name', prediction_data_document['xml_file_name'])

    @mongomock.patch(servers=['mongodb://mongo_information_extraction:27017'])
    def test_get_suggestions(self):
        tenant = "example_tenant_name"
        extraction_name = "prediction_property_name"

        mongo_client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')

        json_data = [{'tenant': 'wrong tenant',
                      'extraction_name': extraction_name,
                      'xml_file_name': "one_file_name",
                      'text': "one_text_predicted",
                      'segment_text': "one_segment_text",
                      }, {'tenant': tenant,
                          'extraction_name': extraction_name,
                          'xml_file_name': "one_file_name",
                          'text': "one_text_predicted",
                          'segment_text': "one_segment_text",
                          }, {'tenant': tenant,
                              'extraction_name': extraction_name,
                              'xml_file_name': "other_file_name",
                              'text': "other_text_predicted",
                              'segment_text': "other_segment_text",
                              }, {'tenant': tenant,
                                  'extraction_name': 'wrong extraction name',
                                  'xml_file_name': "other_file_name",
                                  'text': "other_text_predicted",
                                  'segment_text': "other_segment_text",
                                  }]

        mongo_client.pdf_information_extraction.suggestions.insert_many(json_data)

        response = client.get(f"/get_suggestions/{tenant}/{extraction_name}")
        suggestions = json.loads(response.json())

        self.assertEqual(200, response.status_code)
        self.assertEqual(2, len(suggestions))

        self.assertEqual({tenant}, {x['tenant'] for x in suggestions})
        self.assertEqual({extraction_name}, {x['extraction_name'] for x in suggestions})

        self.assertEqual("one_file_name", suggestions[0]['xml_file_name'])
        self.assertEqual("one_segment_text", suggestions[0]['segment_text'])
        self.assertEqual("one_text_predicted", suggestions[0]['text'])

        self.assertEqual("other_file_name", suggestions[1]['xml_file_name'])
        self.assertEqual("other_segment_text", suggestions[1]['segment_text'])
        self.assertEqual("other_text_predicted", suggestions[1]['text'])

    @mongomock.patch(servers=['mongodb://mongo_information_extraction:27017'])
    def test_should_remove_suggestions_when_returned(self):
        tenant = "example_tenant_name"
        extraction_name = "prediction_property_name"

        mongo_client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')

        json_data = [{'tenant': 'wrong tenant',
                      'extraction_name': extraction_name,
                      'xml_file_name': "one_file_name",
                      'text': "one_text_predicted",
                      'segment_text': "one_segment_text",
                      }, {'tenant': tenant,
                          'extraction_name': extraction_name,
                          'xml_file_name': "one_file_name",
                          'text': "one_text_predicted",
                          'segment_text': "one_segment_text",
                          }, {'tenant': tenant,
                              'extraction_name': extraction_name,
                              'xml_file_name': "other_file_name",
                              'text': "other_text_predicted",
                              'segment_text': "other_segment_text",
                              }, {'tenant': tenant,
                                  'extraction_name': 'wrong extraction name',
                                  'xml_file_name': "other_file_name",
                                  'text': "other_text_predicted",
                                  'segment_text': "other_segment_text",
                                  }]

        mongo_client.pdf_information_extraction.suggestions.insert_many(json_data)

        client.get(f"/get_suggestions/{tenant}/{extraction_name}")

        suggestions: List[Suggestion] = list()

        for document in mongo_client.pdf_information_extraction.suggestions.find():
            suggestions.append(Suggestion(**document))

        self.assertEqual(2, len(suggestions))
        self.assertEqual('wrong tenant', suggestions[0].tenant)
        self.assertEqual('wrong extraction name', suggestions[1].extraction_name)

    @mongomock.patch(servers=['mongodb://mongo_information_extraction:27017'])
    def test_get_suggestions_when_no_suggestions(self):
        tenant = "example_tenant_name"
        extraction_name = "prediction_property_name"

        response = client.get(f"/get_suggestions/{tenant}/{extraction_name}")
        suggestions = json.loads(response.json())

        self.assertEqual(200, response.status_code)
        self.assertEqual(0, len(suggestions))

    @mongomock.patch(servers=['mongodb://mongo_information_extraction:27017'])
    def test_post_calculate_suggestions(self):
        tenant = "example_tenant_name"
        extraction_name = "extraction_name"

        tenant_url = tenant.replace('_', '%20')
        extraction_name_url = extraction_name.replace('_', '%20')

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

        predict_data_json = {"xml_file_name": "test.xml",
                             "extraction_name": extraction_name,
                             "tenant": tenant,
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": [],
                             }

        labeled_data_json = {"xml_file_name": "test.xml",
                             "extraction_name": extraction_name,
                             "tenant": tenant,
                             "language_iso": "en",
                             "label_text": "text",
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": [],
                             "label_segments_boxes": [{"left": 124, "top": 48, "width": 83, "height": 13,
                                                       "page_number": 1}]
                             }

        with open('DOCKER_VOLUME_PATH/tenant_test/extraction_name/xml_files/test.xml', 'rb') as stream:
            files = {'file': stream}
            client.post(f"/xml_file/{tenant_url}/{extraction_name_url}", files=files)

        client.post("/prediction_data", json=predict_data_json)
        client.post("/labeled_data", json=labeled_data_json)

        calculate_result = client.post(f"/calculate_suggestions/{tenant_url}/{extraction_name_url}")

        response = client.get(f"/get_suggestions/{tenant_url}/{extraction_name_url}")
        suggestions = json.loads(response.json())

        suggestion_1 = Suggestion(**suggestions[0])

        self.assertEqual(200, calculate_result.status_code)
        self.assertEqual(2, len(suggestions))

        self.assertEqual(tenant, suggestion_1.tenant)
        self.assertEqual(extraction_name, suggestion_1.extraction_name)
        self.assertEqual("test.xml", suggestion_1.xml_file_name)
        self.assertEqual("United Nations", suggestion_1.segment_text)
        self.assertEqual("United Nations", suggestion_1.text)

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

    @mongomock.patch(servers=['mongodb://mongo_information_extraction:27017'])
    def test_post_calculate_suggestions_when_no_label_data(self):
        tenant = "example_tenant_name"
        extraction_name = "extraction_name"

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

        calculate_result = client.post(f"/calculate_suggestions/{tenant}/{extraction_name}")

        self.assertEqual(200, calculate_result.status_code)

    def test_create_model(self):
        tenant = 'semantic_tenant_to_be_removed_1'
        extraction_name = 'extraction_name'

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

        semantic_information_extraction = SemanticInformationExtraction(tenant=tenant, extraction_name=extraction_name)

        semantic_information_data = [SemanticExtractionData(text="one", segment_text="one two", language_iso="en")]
        semantic_information_extraction.create_model(semantic_information_data)

        self.assertTrue(os.path.exists(f'{DOCKER_VOLUME_PATH}/{tenant}/{extraction_name}/semantic_model/best_model'))

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

