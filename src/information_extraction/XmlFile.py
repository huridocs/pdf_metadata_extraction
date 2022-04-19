import os
import pathlib
from collections import defaultdict
from typing import List

from bs4 import BeautifulSoup

from ServiceConfig import ServiceConfig
from data.SegmentationData import SegmentationData
from information_extraction.Font import Font
from information_extraction.PdfFeatures import PdfFeatures
from information_extraction.Segment import Segment
from information_extraction.SegmentTag import SegmentTag


class XmlFile:
    def __init__(self, tenant: str, property_name: str, to_train: bool, xml_file_name: str):
        self.tenant = tenant
        self.property_name = property_name
        self.to_train = to_train
        self.xml_file_name = xml_file_name
        self.xml_file = None
        self.segments = list()
        self.xml_folder_path = XmlFile.get_xml_folder_path(tenant, property_name, to_train)

    def save(self, file: bytes):
        if not os.path.exists(self.xml_folder_path):
            os.makedirs(self.xml_folder_path)

        file_path = pathlib.Path(f"{self.xml_folder_path}/{self.xml_file_name}")
        file_path.write_bytes(file)

    def get_segments(self, segmentation_data: SegmentationData):
        try:
            self.xml_file = pathlib.Path(f"{self.xml_folder_path}/{self.xml_file_name}").read_bytes()
        except FileNotFoundError:
            return []

        if not segmentation_data:
            return []

        self.create_segments(segmentation_data)
        return self.segments

    @staticmethod
    def is_empty(segment_tag: SegmentTag):
        if segment_tag.text.replace(" ", "") == "":
            return True

        return False

    def get_one_tag_segments(self) -> (List[SegmentTag], List[Font]):
        segment_tags = list()
        xml = BeautifulSoup(self.xml_file, "lxml-xml")
        xml_fonts = xml.find_all("TextStyle")
        fonts = Font.from_page_xml_tag(xml_fonts)
        for xml_page in xml.find_all("Page"):
            page_number = int(xml_page["PHYSICAL_IMG_NR"])
            page_width = float(xml_page["WIDTH"])
            page_height = float(xml_page["HEIGHT"])
            for text_line in self.get_text_lines(xml_page):
                segment_tags.append(SegmentTag(text_line, page_width, page_height, page_number, fonts))

        pdf_features = PdfFeatures(segment_tags) if segment_tags else []
        not_empty_segment_tags = [x for x in segment_tags if not self.is_empty(x)]
        one_tag_segments = [Segment(segment_tag, pdf_features) for segment_tag in not_empty_segment_tags]

        return one_tag_segments

    @staticmethod
    def get_text_lines(xml_page):
        page_text_lines = list()
        for text_line in xml_page.find_all("TextLine"):
            if len(list(text_line.find_all("String"))) == 0:
                continue

            page_text_lines.append(text_line)

        return page_text_lines

    def create_segments(self, segmentation_data: SegmentationData):
        box_segments_to_merge = defaultdict(list)

        for segment in self.get_one_tag_segments():
            index = segment.intersects_with_boxes(
                segmentation_data.page_width,
                segmentation_data.page_height,
                segmentation_data.xml_segments_boxes,
            )

            if index is not None:
                box_segments_to_merge[index].append(segment)
            else:
                box_segments_to_merge[segment].append(segment)

        self.segments = [Segment.merge(segments_to_merge) for segments_to_merge in box_segments_to_merge.values()]

        for segment in self.segments:
            segment.set_ml_label(
                segmentation_data.page_width,
                segmentation_data.page_height,
                segmentation_data.label_segments_boxes,
            )

    @staticmethod
    def get_xml_folder_path(tenant: str, property_name: str, to_train: bool) -> str:
        service_config = ServiceConfig()
        xml_folder_path = f"{service_config.docker_volume_path}/{tenant}/{property_name}"
        if to_train:
            xml_folder_path += f"/xml_to_train"
        else:
            xml_folder_path += f"/xml_to_predict"

        return xml_folder_path
