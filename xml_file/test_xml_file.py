import os
from unittest import TestCase

from xml_file.XmlFile import XmlFile


class TestXmlFile(TestCase):
    def test_save(self):
        with open('../test_pdf/test.pdf', 'rb') as file:
            XmlFile(filename='test.pdf', tenant='tenant_one').save(file=file.read())

        self.assertTrue(os.path.exists('../docker_volume/labeled_xmls/tenant_one/test.pdf'))
        os.remove('../docker_volume/labeled_xmls/tenant_one/test.pdf')
        os.rmdir('../docker_volume/labeled_xmls/tenant_one')