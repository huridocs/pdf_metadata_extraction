import os
from unittest import TestCase

import mongomock
import pymongo

from segment_predictor.SegmentPredictor import SegmentPredictor


class TestSegmentPredictor(TestCase):
    @mongomock.patch(servers=['mongodb://mongo:27017'])
    def test_create_model(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo:27017')

        json_data = {"xml_file_name": "test.xml",
                     "extraction_name": "extraction_name",
                     "tenant": "tenant_test",
                     "label_text": "text",
                     "page_width": 612,
                     "page_height": 792,
                     "xml_segments_boxes": [
                         {"left": 495.1, "top": 42.6323, "width": 56.96199999999999, "height": 18.2164,
                          "page_number": 1},
                         {"left": 123.38, "top": 48.1103, "width": 82.9812, "height": 12.7624,
                          "page_number": 1},
                         {"left": 123.38, "top": 72.8529, "width": 148.656, "height": 17.895700000000005,
                          "page_number": 1},
                         {"left": 123.38, "top": 245.184, "width": 317.406, "height": 27.5377,
                          "page_number": 1},
                     ],
                     "label_segments_boxes": [{"left": 125, "top": 247, "width": 319, "height": 29, "page_number": 1}]
                     }

        mongo_client.pdf_information_extraction.labeleddata.insert_one(json_data)

        segment_predictor = SegmentPredictor("extraction_name", "tenant_test")
        segment_predictor.create_model()

        self.assertTrue(os.path.exists('../../docker_volume/segment_test/extraction_name/model/model.model'))
        self.assertFalse(os.path.exists('../../docker_volume/segment_test/extraction_name/xml_files'))


