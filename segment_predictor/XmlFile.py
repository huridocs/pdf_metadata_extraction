import os
import pathlib
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List

import pymongo
from bs4 import BeautifulSoup

from data.LabeledData import LabeledData
from segment_predictor.Font import Font
from segment_predictor.PdfFeatures import PdfFeatures
from segment_predictor.Segment import Segment
from segment_predictor.SegmentTag import SegmentTag


class XmlFile:
    def __init__(self, tenant: str, extraction_name: str, file_name: str = ''):
        self.file_name = file_name
        self.tenant = tenant
        self.extraction_name = extraction_name
        self.xml_folder_path = XmlFile.get_xml_folder_path(tenant, extraction_name)
        self.file_path = pathlib.Path(f'{self.xml_folder_path}/{self.file_name}')
        self.xml_file = None
        self.segments = list()

    def save(self, file: bytes):
        self.xml_file = file

        if not os.path.exists(self.xml_folder_path):
            os.makedirs(self.xml_folder_path)

        self.file_path.write_bytes(self.xml_file)

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
            index = segment.intersects_with_boxes(labeled_data.xml_segments_boxes)

            if index is not None:
                box_segments_to_merge[index].append(segment)
            else:
                box_segments_to_merge[segment].append(segment)

        self.segments = [Segment.merge(segments_to_merge) for segments_to_merge in box_segments_to_merge.values()]

        for segment in self.segments:
            segment.set_ml_label(labeled_data.label_segments_boxes)

    @staticmethod
    def get_segments(labeled_data: LabeledData):
        xml_file = XmlFile(file_name=labeled_data.xml_file_name, tenant=labeled_data.tenant,
                           extraction_name=labeled_data.extraction_name)
        xml_file.set_segments(labeled_data)
        return xml_file.segments

    @staticmethod
    def get_xml_folder_path(tenant: str, extraction_name: str):
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        return f'{path.parent.absolute()}/docker_volume/{tenant}/{extraction_name}/xml_files'
