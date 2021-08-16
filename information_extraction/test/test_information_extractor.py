import os
import shutil
from typing import List
from unittest import TestCase

import mongomock
import pymongo

from data.SegmentBox import SegmentBox
from data.Suggestion import Suggestion
from information_extraction.InformationExtraction import InformationExtraction

DOCKER_VOLUME_PATH = f'{os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))}/docker_volume'


class TestInformationExtractor(TestCase):
    @mongomock.patch(servers=['mongodb://mongo_information_extraction:27017'])
    def test_create_model(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')
        json_data = {"xml_file_name": "test.xml",
                     "extraction_name": "extraction_name",
                     "tenant": "segment_test",
                     "label_text": "text",
                     "page_width": 612,
                     "page_height": 792,
                     "xml_segments_boxes": [],
                     "label_segments_boxes": [{"left": 123, "top": 48, "width": 83, "height": 12, "page_number": 1}]
                     }

        mongo_client.pdf_information_extraction.labeleddata.insert_one(json_data)

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/segment_test', ignore_errors=True)

        os.makedirs(f'{DOCKER_VOLUME_PATH}/segment_test/extraction_name/xml_files')
        shutil.copy(f'{DOCKER_VOLUME_PATH}/tenant_test/extraction_name/xml_files/test.xml',
                    f'{DOCKER_VOLUME_PATH}/segment_test/extraction_name/xml_files/test.xml')

        information_extraction = InformationExtraction("segment_test", "extraction_name")
        information_extraction.create_models()

        self.assertTrue(
            os.path.exists(f'{DOCKER_VOLUME_PATH}/segment_test/extraction_name/segment_predictor_model/model.model'))
        self.assertTrue(os.path.exists(f'{DOCKER_VOLUME_PATH}/segment_test/extraction_name/xml_files'))

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/segment_test')

    @mongomock.patch(servers=['mongodb://mongo_information_extraction:27017'])
    def test_create_model_no_xml(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')
        json_data = {"xml_file_name": "test.xml",
                     "extraction_name": "extraction_name",
                     "tenant": "segment_test",
                     "label_text": "text",
                     "page_width": 612,
                     "page_height": 792,
                     "xml_segments_boxes": [],
                     "label_segments_boxes": [{"left": 125, "top": 247, "width": 319, "height": 29, "page_number": 1}]
                     }

        mongo_client.pdf_information_extraction.labeleddata.insert_one(json_data)

        information_extraction = InformationExtraction("segment_test", "extraction_name")
        information_extraction.create_models()

        self.assertFalse(os.path.exists(f'{DOCKER_VOLUME_PATH}/segment_test/extraction_name/xml_files'))
        self.assertFalse(
            os.path.exists(f'{DOCKER_VOLUME_PATH}/segment_test/extraction_name/segment_predictor_model/model.model'))

    @mongomock.patch(servers=['mongodb://mongo_information_extraction:27017'])
    def test_create_model_wrong_tenant(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')
        json_data = {"xml_file_name": "test.xml",
                     "extraction_name": "extraction_name",
                     "tenant": "other_tenant_name",
                     "label_text": "text",
                     "page_width": 612,
                     "page_height": 792,
                     "xml_segments_boxes": [],
                     "label_segments_boxes": [{"left": 125, "top": 247, "width": 319, "height": 29, "page_number": 1}]
                     }

        mongo_client.pdf_information_extraction.labeleddata.insert_one(json_data)

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/segment_test', ignore_errors=True)

        os.makedirs(f'{DOCKER_VOLUME_PATH}/segment_test/extraction_name/xml_files')
        shutil.copy(f'{DOCKER_VOLUME_PATH}/tenant_test/extraction_name/xml_files/test.xml',
                    f'{DOCKER_VOLUME_PATH}/segment_test/extraction_name/xml_files/test.xml')

        information_extraction = InformationExtraction("segment_test", "extraction_name")
        information_extraction.create_models()

        self.assertTrue(os.path.exists(f'{DOCKER_VOLUME_PATH}/segment_test/extraction_name/xml_files'))
        self.assertFalse(
            os.path.exists(f'{DOCKER_VOLUME_PATH}/segment_test/extraction_name/segment_predictor_model/model.model'))

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/segment_test')

    @mongomock.patch(servers=['mongodb://mongo_information_extraction:27017'])
    def test_create_model_wrong_extraction_name(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')
        json_data = {"xml_file_name": "test.xml",
                     "extraction_name": "other_extraction_name",
                     "tenant": "segment_test",
                     "label_text": "text",
                     "page_width": 612,
                     "page_height": 792,
                     "xml_segments_boxes": [],
                     "label_segments_boxes": [{"left": 125, "top": 247, "width": 319, "height": 29, "page_number": 1}]
                     }

        mongo_client.pdf_information_extraction.labeleddata.insert_one(json_data)

        try:
            shutil.rmtree(f'{DOCKER_VOLUME_PATH}/segment_test')
        except FileNotFoundError:
            pass

        os.makedirs(f'{DOCKER_VOLUME_PATH}/segment_test/extraction_name/xml_files')
        shutil.copy(f'{DOCKER_VOLUME_PATH}/tenant_test/extraction_name/xml_files/test.xml',
                    f'{DOCKER_VOLUME_PATH}/segment_test/extraction_name/xml_files/test.xml')

        information_extraction = InformationExtraction("segment_test", "extraction_name")
        information_extraction.create_models()

        self.assertTrue(os.path.exists(f'{DOCKER_VOLUME_PATH}/segment_test/extraction_name/xml_files'))
        self.assertFalse(
            os.path.exists(f'{DOCKER_VOLUME_PATH}/segment_test/extraction_name/segment_predictor_model/model.model'))

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/segment_test')

    @mongomock.patch(servers=['mongodb://mongo_information_extraction:27017'])
    def test_get_suggestions(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')

        tenant = "tenant_to_be_removed"
        extraction_name = "extraction_name"

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)
        shutil.copytree(f'{DOCKER_VOLUME_PATH}/tenant_test', f'{DOCKER_VOLUME_PATH}/{tenant}')

        to_predict_json = {"xml_file_name": "test.xml",
                           "extraction_name": extraction_name,
                           "tenant": tenant,
                           "page_width": 612,
                           "page_height": 792,
                           "xml_segments_boxes": []
                           }

        labeled_data_json = {"xml_file_name": "test.xml",
                             "extraction_name": extraction_name,
                             "tenant": tenant,
                             "label_text": "text",
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": [],
                             "label_segments_boxes": [{"left": 124, "top": 48, "width": 83, "height": 13,
                                                       "page_number": 1}]
                             }

        mongo_client.pdf_information_extraction.labeleddata.insert_one(labeled_data_json)
        mongo_client.pdf_information_extraction.predictiondata.insert_one(to_predict_json)

        information_extraction = InformationExtraction(tenant, extraction_name)
        information_extraction.calculate_suggestions()

        suggestions: List[Suggestion] = list()
        find_filter = {"extraction_name": extraction_name, "tenant": tenant}
        for document in mongo_client.pdf_information_extraction.suggestions.find(find_filter, no_cursor_timeout=True):
            suggestions.append(Suggestion(**document))

        suggestion_1 = suggestions[0]
        suggestion_2 = suggestions[1]

        self.assertEqual(2, len(suggestions))

        self.assertEqual(tenant, suggestion_1.tenant)
        self.assertEqual(extraction_name, suggestion_1.extraction_name)
        self.assertEqual("test.xml", suggestion_1.xml_file_name)
        self.assertEqual("United Nations", suggestion_1.segment_text)
        self.assertEqual("United Nations", suggestion_1.text)

        self.assertEqual(tenant, suggestion_2.tenant)
        self.assertEqual(extraction_name, suggestion_2.extraction_name)
        self.assertEqual("test.xml", suggestion_2.xml_file_name)
        self.assertEqual("United Nations", suggestion_2.segment_text)
        self.assertEqual("United Nations", suggestion_2.text)

        self.assertIsNone(mongo_client.pdf_information_extraction.labeleddata.find_one())
        self.assertIsNone(mongo_client.pdf_information_extraction.predictiondata.find_one())

        self.assertFalse(os.path.exists(f'{DOCKER_VOLUME_PATH}/{tenant}/{extraction_name}/xml_files'))

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

    @mongomock.patch(servers=['mongodb://mongo_information_extraction:27017'])
    def test_get_suggestions_page_2(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')

        tenant = "tenant_to_be_removed"
        extraction_name = "extraction_name"

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)
        shutil.copytree(f'{DOCKER_VOLUME_PATH}/tenant_test', f'{DOCKER_VOLUME_PATH}/{tenant}')

        to_predict_json = {"xml_file_name": "test.xml",
                           "extraction_name": extraction_name,
                           "tenant": tenant,
                           "page_width": 306,
                           "page_height": 396,
                           "xml_segments_boxes": [SegmentBox(left=0, top=75, width=306, height=35,
                                                             page_number=2).dict()],
                           }

        labeled_data_json = {"xml_file_name": "test.xml",
                             "extraction_name": extraction_name,
                             "tenant": tenant,
                             "label_text": "text",
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": [SegmentBox(left=0, top=130, width=612, height=70,
                                                               page_number=2).dict()],
                             "label_segments_boxes": [
                                 SegmentBox(left=300, top=150, width=5, height=5, page_number=2).dict()]
                             }

        mongo_client.pdf_information_extraction.labeleddata.insert_one(labeled_data_json)
        mongo_client.pdf_information_extraction.predictiondata.insert_one(to_predict_json)

        information_extraction = InformationExtraction(tenant, extraction_name)
        information_extraction.calculate_suggestions()

        suggestions: List[Suggestion] = list()
        find_filter = {"extraction_name": extraction_name, "tenant": tenant}
        for document in mongo_client.pdf_information_extraction.suggestions.find(find_filter, no_cursor_timeout=True):
            suggestions.append(Suggestion(**document))

        suggestion_1 = suggestions[0]
        suggestion_2 = suggestions[1]

        self.assertEqual(2, len(suggestions))

        self.assertEqual(tenant, suggestion_1.tenant)
        self.assertEqual(extraction_name, suggestion_1.extraction_name)
        self.assertEqual("test.xml", suggestion_1.xml_file_name)
        self.assertTrue('In accordance with paragraph' in suggestion_1.segment_text)
        self.assertTrue('every four years' in suggestion_1.text)

        self.assertEqual(tenant, suggestion_2.tenant)
        self.assertEqual(extraction_name, suggestion_2.extraction_name)
        self.assertEqual("test.xml", suggestion_2.xml_file_name)
        self.assertTrue('In accordance with paragraph' in suggestion_1.segment_text)
        self.assertTrue('every four years' in suggestion_1.text)

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

    @mongomock.patch(servers=['mongodb://mongo_information_extraction:27017'])
    def test_get_semantic_suggestions(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')

        tenant = "tenant_to_be_removed"
        extraction_name = "extraction_name"

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)
        shutil.copytree(f'{DOCKER_VOLUME_PATH}/tenant_test', f'{DOCKER_VOLUME_PATH}/{tenant}')

        to_predict_json = {"xml_file_name": "test.xml",
                           "extraction_name": extraction_name,
                           "tenant": tenant,
                           "page_width": 612,
                           "page_height": 792,
                           "xml_segments_boxes": [],
                           }
        mongo_client.pdf_information_extraction.predictiondata.insert_one(to_predict_json)

        for i in range(7):
            labeled_data_json = {"xml_file_name": "test.xml",
                                 "extraction_name": extraction_name,
                                 "tenant": tenant,
                                 "label_text": "English",
                                 "page_width": 612,
                                 "page_height": 792,
                                 "xml_segments_boxes": [],
                                 "label_segments_boxes": [
                                     SegmentBox(left=397, top=115, width=74, height=9, page_number=1).dict()]
                                 }

            mongo_client.pdf_information_extraction.labeleddata.insert_one(labeled_data_json)

        information_extraction = InformationExtraction(tenant, extraction_name)
        information_extraction.calculate_suggestions()

        suggestions: List[Suggestion] = list()
        find_filter = {"extraction_name": extraction_name, "tenant": tenant}
        for document in mongo_client.pdf_information_extraction.suggestions.find(find_filter, no_cursor_timeout=True):
            suggestions.append(Suggestion(**document))

        self.assertEqual(8, len(suggestions))

        self.assertEqual({tenant}, {x.tenant for x in suggestions})
        self.assertEqual({extraction_name}, {x.extraction_name for x in suggestions})
        self.assertEqual({extraction_name}, {x.extraction_name for x in suggestions})
        self.assertEqual({"test.xml"}, {x.xml_file_name for x in suggestions})
        self.assertEqual({"Original: English"}, {x.segment_text for x in suggestions})
        self.assertEqual({"English"}, {x.text for x in suggestions})

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)
