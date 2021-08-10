import os
from unittest import TestCase
import shutil

import mongomock
import pymongo

from segment_predictor.XmlFile import XmlFile


class TestXmlFile(TestCase):
    tenant = "tenant_test"

    def test_save(self):
        with open('../../docker_volume/tenant_test/xml_files/test.xml', 'rb') as file:
            XmlFile(file_name='test.xml', tenant='tenant_one').save(file=file.read())

        self.assertTrue(os.path.exists('../../docker_volume/tenant_one/xml_files/test.xml'))
        os.remove('../../docker_volume/tenant_one/xml_files/test.xml')
        os.rmdir('../../docker_volume/tenant_one/xml_files')
        os.rmdir('../../docker_volume/tenant_one')

    @mongomock.patch(servers=['mongodb://mongo:27017'])
    def test_set_segments(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo:27017')

        json_data = {"xml_file_name": "test.xml",
                     "extraction_name": "extraction_name",
                     "tenant": TestXmlFile.tenant,
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

        xml_file = XmlFile(file_name='test.xml', tenant=TestXmlFile.tenant)
        xml_file.set_segments(extraction_name='extraction_name')

        self.assertEqual(612, xml_file.segments[0].page_width)
        self.assertEqual(792, xml_file.segments[0].page_height)
        self.assertEqual(1, len([segment for segment in xml_file.segments if segment.ml_class_label == 1]))
        self.assertEqual('United Nations', xml_file.segments[0].text_content)
        self.assertEqual('A/INF/76/1', xml_file.segments[1].text_content)
        self.assertEqual('General Assembly', xml_file.segments[2].text_content)
        self.assertEqual(
            'Opening dates of forthcoming regular sessions of the General Assembly and of the general debate',
            [segment for segment in xml_file.segments if segment.ml_class_label == 1][0].text_content)

    @mongomock.patch(servers=['mongodb://mongo:27017'])
    def test_set_segments_page_2(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo:27017')

        json_data = {"xml_file_name": "test.xml",
                     "extraction_name": "extraction_name",
                     "tenant": TestXmlFile.tenant,
                     "label_text": "text",
                     "page_width": 612,
                     "page_height": 792,
                     "xml_segments_boxes": [
                         {"left": 0, "top": 130, "width": 612, "height": 70,
                          "page_number": 2},
                     ],
                     "label_segments_boxes": [{"left": 300, "top": 150, "width": 5, "height": 5, "page_number": 2}]
                     }

        mongo_client.pdf_information_extraction.labeleddata.insert_one(json_data)

        xml_file = XmlFile(file_name='test.xml', tenant=TestXmlFile.tenant)
        xml_file.set_segments(extraction_name='extraction_name')
        labeled_segments = [segment for segment in xml_file.segments if segment.ml_class_label == 1]

        self.assertEqual(612, xml_file.segments[0].page_width)
        self.assertEqual(792, xml_file.segments[0].page_height)
        self.assertEqual(1, len(labeled_segments))
        self.assertTrue('In accordance with paragraph' in labeled_segments[0].text_content)

    @mongomock.patch(servers=['mongodb://mongo:27017'])
    def test_set_segments_when_no_file(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo:27017')

        json_data = {"xml_file_name": "test.xml",
                     "extraction_name": "extraction_name",
                     "tenant": 'non-existent_tenant',
                     "label_text": "text",
                     "page_width": 612,
                     "page_height": 792,
                     "xml_segments_boxes": [],
                     "label_segments_boxes": [{"left": 125, "top": 247, "width": 319, "height": 29, "page_number": 1}]
                     }

        mongo_client.pdf_information_extraction.labeleddata.insert_one(json_data)

        xml_file = XmlFile(file_name='test.xml', tenant=TestXmlFile.tenant)
        xml_file.set_segments(extraction_name='extraction_name')

        self.assertEqual(0, len(xml_file.segments))

    @mongomock.patch(servers=['mongodb://mongo:27017'])
    def test_set_segments_when_no_data_base_entry(self):
        xml_file = XmlFile(file_name='test.xml', tenant=TestXmlFile.tenant)
        xml_file.set_segments(extraction_name='extraction_name')

        self.assertEqual(0, len(xml_file.segments))

