import os
from unittest import TestCase
import shutil

import mongomock
import pymongo

from segment_predictor.XmlFile import XmlFile


class TestXmlFile(TestCase):
    tenant = "test_segments_tenant"

    def test_save(self):
        with open('../../test_xml/test.xml', 'rb') as file:
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

        os.mkdir(f'../../docker_volume/{TestXmlFile.tenant}')
        os.mkdir(f'../../docker_volume/{TestXmlFile.tenant}/xml_files')
        shutil.copyfile(f'../../test_xml/test.xml', f'../../docker_volume/{TestXmlFile.tenant}/xml_files/test.xml')

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

        os.remove(f'../../docker_volume/{TestXmlFile.tenant}/xml_files/test.xml')
        os.rmdir(f'../../docker_volume/{TestXmlFile.tenant}/xml_files')
        os.rmdir(f'../../docker_volume/{TestXmlFile.tenant}')

    @mongomock.patch(servers=['mongodb://mongo:27017'])
    def test_set_segments_when_no_file(self):
        mongo_client = pymongo.MongoClient('mongodb://mongo:27017')

        json_data = {"xml_file_name": "test.xml",
                     "extraction_name": "extraction_name",
                     "tenant": TestXmlFile.tenant,
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
    def test_set_segments_when_data_base_entry(self):
        os.mkdir(f'../../docker_volume/{TestXmlFile.tenant}')
        os.mkdir(f'../../docker_volume/{TestXmlFile.tenant}/xml_files')
        shutil.copyfile(f'../../test_xml/test.xml', f'../../docker_volume/{TestXmlFile.tenant}/xml_files/test.xml')

        xml_file = XmlFile(file_name='test.xml', tenant=TestXmlFile.tenant)
        xml_file.set_segments(extraction_name='extraction_name')

        self.assertEqual(0, len(xml_file.segments))

        os.remove(f'../../docker_volume/{TestXmlFile.tenant}/xml_files/test.xml')
        os.rmdir(f'../../docker_volume/{TestXmlFile.tenant}/xml_files')
        os.rmdir(f'../../docker_volume/{TestXmlFile.tenant}')

