import os
import pathlib
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List

import pymongo
from bs4 import BeautifulSoup

from data.LabeledData import LabeledData
from information_extraction.Font import Font
from information_extraction.PdfFeatures import PdfFeatures
from information_extraction.Segment import Segment
from information_extraction.SegmentTag import SegmentTag


class XmlFile:
    def __init__(self, tenant: str, template: str, property_name: str, file_name: str = ''):
        self.tenant = tenant
        self.template = template
        self.property_name = property_name
        self.file_name = file_name
        self.xml_file = None
        self.segments = list()

    def save_as_to_train(self, file: bytes):
        if not os.path.exists(self.get_to_train_xml_folder_path()):
            os.makedirs(self.get_to_train_xml_folder_path())

        file_path = pathlib.Path(f'{self.get_to_train_xml_folder_path()}/{self.file_name}')
        file_path.write_bytes(file)

    def save_as_to_predict(self, file):
        if not os.path.exists(self.get_to_predict_xml_folder_path()):
            os.makedirs(self.get_to_predict_xml_folder_path())

        file_path = pathlib.Path(f'{self.get_to_predict_xml_folder_path()}/{self.file_name}')
        file_path.write_bytes(file)

    def set_segments(self, labeled_data: LabeledData):
        try:
            self.xml_file = self.file_path.read_bytes()
        except FileNotFoundError:
            return

        if not labeled_data:
            return

        one_tag_segments = self.get_one_tag_segments(BeautifulSoup(self.xml_file, 'lxml-xml'))
        self.create_segments_from_labeled_data(one_tag_segments, labeled_data)

    def get_one_tag_segments(self, xml: BeautifulSoup) -> (List[SegmentTag], List[Font]):
        segment_tags = list()
        xml_fonts = xml.find_all('TextStyle')
        fonts = Font.from_page_xml_tag(xml_fonts)
        for xml_page in xml.find_all('Page'):
            page_number = int(xml_page['PHYSICAL_IMG_NR'])
            page_width = float(xml_page['WIDTH'])
            page_height = float(xml_page['HEIGHT'])
            for text_line in self.get_text_lines(xml_page):
                segment_tags.append(SegmentTag(text_line, page_width, page_height, page_number, fonts))

        pdf_features = PdfFeatures(segment_tags)
        one_tag_segments = [Segment(segment_tag, pdf_features) for segment_tag in segment_tags]

        return one_tag_segments

    @staticmethod
    def get_text_lines(xml_page):
        page_text_lines = list()
        for text_line in xml_page.find_all('TextLine'):
            if len(list(text_line.find_all('String'))) == 0:
                continue

            page_text_lines.append(text_line)

        return page_text_lines

    def create_segments_from_labeled_data(self, one_tag_segments: List[Segment], labeled_data: LabeledData):
        box_segments_to_merge = defaultdict(list)
        for segment in one_tag_segments:
            index = segment.intersects_with_boxes(labeled_data.page_width, labeled_data.page_height,
                                                  labeled_data.xml_segments_boxes)

            if index is not None:
                box_segments_to_merge[index].append(segment)
            else:
                box_segments_to_merge[segment].append(segment)

        self.segments = [Segment.merge(segments_to_merge) for segments_to_merge in box_segments_to_merge.values()]

        for segment in self.segments:
            segment.set_ml_label(labeled_data.page_width, labeled_data.page_height, labeled_data.label_segments_boxes)

    def get_to_train_xml_folder_path(self):
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        training_xml_folder_path = f'{path.parent.absolute()}/docker_volume'
        training_xml_folder_path += f'/{self.tenant}/{self.template}/{self.property_name}'
        training_xml_folder_path += f'/to_train_xml'
        return training_xml_folder_path

    def get_to_predict_xml_folder_path(self):
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        training_xml_folder_path = f'{path.parent.absolute()}/docker_volume'
        training_xml_folder_path += f'/{self.tenant}/{self.template}/{self.property_name}'
        training_xml_folder_path += f'/to_predict_xml'
        return training_xml_folder_path

    @staticmethod
    def get_segments(labeled_data: LabeledData):
        xml_file = XmlFile(file_name=labeled_data.xml_file_name, tenant=labeled_data.tenant,
                           extraction_name=labeled_data.property_name)
        xml_file.set_segments(labeled_data)
        return xml_file.segments

    @staticmethod
    def remove_files(tenant, extraction_name):
        shutil.rmtree(XmlFile.get_to_train_xml_folder_path(tenant, extraction_name), ignore_errors=True)

