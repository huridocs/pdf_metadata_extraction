import os
import pathlib
from collections import defaultdict
from pathlib import Path
from typing import List

import pymongo
from bs4 import BeautifulSoup

from data.LabeledData import LabeledData
from segment_predictor.Font import Font
from segment_predictor.PDFFeatures import PDFFeatures
from segment_predictor.Segment import Segment
from segment_predictor.SegmentTag import SegmentTag


class XmlFile:
    def __init__(self, file_name: str, tenant: str):
        path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.root_folder = path.parent.absolute()
        self.tenant = tenant
        self.file_name = file_name
        self.file_path = pathlib.Path(f'{self.root_folder}/docker_volume/{self.tenant}/xml_files/{self.file_name}')
        self.xml_file = None
        self.segments = list()
        self.extraction_name = None

    def save(self, file: bytes):
        self.xml_file = file
        if not os.path.exists(f'{self.root_folder}/docker_volume/{self.tenant}'):
            os.mkdir(f'{self.root_folder}/docker_volume/{self.tenant}')

        if not os.path.exists(self.file_path.parents[0]):
            os.mkdir(self.file_path.parents[0])

        self.file_path.write_bytes(self.xml_file)

    def set_segments(self, extraction_name):
        try:
            self.xml_file = self.file_path.read_bytes()
        except FileNotFoundError:
            return

        self.extraction_name = extraction_name
        labeled_data = self.get_labeled_data_from_database()

        if not labeled_data:
            return

        segment_tags = self.get_segment_tags(BeautifulSoup(self.xml_file, 'lxml-xml'))
        pdf_features = PDFFeatures(segment_tags)
        one_tag_segments = [Segment(segment_tag, pdf_features) for segment_tag in segment_tags]

        self.load_segments(one_tag_segments, labeled_data)

    def get_segment_tags(self, xml: BeautifulSoup) -> (List[SegmentTag], List[Font]):
        segment_tags = list()
        xml_fonts = xml.find_all('TextStyle')
        fonts = Font.from_page_xml_tag(xml_fonts)
        for xml_page in xml.find_all('Page'):
            page_number = int(xml_page['PHYSICAL_IMG_NR'])
            page_width = float(xml_page['WIDTH'])
            page_height = float(xml_page['HEIGHT'])
            for text_line in self.get_text_lines(xml_page):
                segment_tags.append(SegmentTag(text_line, page_width, page_height, page_number, fonts))

        return segment_tags

    @staticmethod
    def get_text_lines(xml_page):
        page_text_lines = list()
        for text_line in xml_page.find_all('TextLine'):
            if len(list(text_line.find_all('String'))) == 0:
                continue

            page_text_lines.append(text_line)

        return page_text_lines

    def get_labeled_data_from_database(self):
        client = pymongo.MongoClient('mongodb://mongo:27017')
        pdf_information_extraction_db = client['pdf_information_extraction']

        find_filter = {"extraction_name": self.extraction_name, "tenant": self.tenant, "xml_file_name": self.file_name}
        document = pdf_information_extraction_db.labeleddata.find_one(find_filter)

        if not document:
            return

        return LabeledData(**document)

    def load_segments(self, one_tag_segments: List[Segment], labeled_data: LabeledData):
        box_segments_to_merge = defaultdict(list)
        for segment in one_tag_segments:
            index = segment.intersects_with_boxes(labeled_data.xml_segments_boxes)

            if index:
                box_segments_to_merge[index].append(segment)
            else:
                box_segments_to_merge[segment].append(segment)

        self.segments = [Segment.merge(segments_to_merge) for segments_to_merge in box_segments_to_merge.values()]

        for segment in self.segments:
            segment.set_ml_label(labeled_data.label_segments_boxes)
