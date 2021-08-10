import os
import shutil
from unittest import TestCase

import mongomock
import pymongo

from segment_predictor.SegmentPredictor import SegmentPredictor


class TestSegmentPredictor(TestCase):

    @mongomock.patch(servers=['mongodb://mongo:27017'])
    def test_create_model(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo:27017')
        root_path = '../../docker_volume'
        json_data = {"xml_file_name": "test.xml",
                     "extraction_name": "extraction_name",
                     "tenant": "segment_test",
                     "label_text": "text",
                     "page_width": 612,
                     "page_height": 792,
                     "xml_segments_boxes": [],
                     "label_segments_boxes": [{"left": 125, "top": 247, "width": 319, "height": 29, "page_number": 1}]
                     }

        json_data_different_file = json_data.copy()
        json_data_different_file["xml_file_name"] = "test_2.xml"
        json_data_different_file["xml_segments_boxes"] = []

        json_data_different_extraction_name = json_data.copy()
        json_data_different_extraction_name["extraction_name"] = "extraction_name_2"
        json_data_different_extraction_name["xml_segments_boxes"] = []

        json_data_different_tenant = json_data.copy()
        json_data_different_tenant["tenant"] = "tenant_2"
        json_data_different_tenant["xml_segments_boxes"] = []

        last_json_data = json_data_different_tenant.copy()

        mongo_client.pdf_information_extraction.labeleddata.insert_one(json_data_different_file)
        mongo_client.pdf_information_extraction.labeleddata.insert_one(json_data_different_extraction_name)
        mongo_client.pdf_information_extraction.labeleddata.insert_one(json_data_different_tenant)
        mongo_client.pdf_information_extraction.labeleddata.insert_one(json_data)
        mongo_client.pdf_information_extraction.labeleddata.insert_one(last_json_data)

        try:
            shutil.rmtree(f'{root_path}/segment_test')
        except FileNotFoundError:
            pass

        os.makedirs(f'{root_path}/segment_test/extraction_name/xml_files')
        shutil.copy(f'{root_path}/tenant_test/extraction_name/xml_files/test.xml',
                    f'{root_path}/segment_test/extraction_name/xml_files/test.xml')

        segment_predictor = SegmentPredictor("segment_test", "extraction_name")
        segment_predictor.create_model()

        self.assertTrue(os.path.exists(f'{root_path}/segment_test/extraction_name/segment_predictor_model/model.model'))
        self.assertFalse(os.path.exists(f'{root_path}/segment_test/extraction_name/xml_files'))

        shutil.rmtree(f'{root_path}/segment_test')


