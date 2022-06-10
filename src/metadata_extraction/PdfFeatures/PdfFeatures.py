import json
import os
import pathlib
from collections import defaultdict
from os.path import exists
from pathlib import Path

from bs4 import BeautifulSoup

from typing import List, Optional, Dict

from data.SegmentationData import SegmentationData
from metadata_extraction.PdfFeatures.PdfAnnotation import PdfAnnotation
from metadata_extraction.PdfFeatures.PdfPage import PdfPage

from metadata_extraction.PdfFeatures.PdfFont import PdfFont

from metadata_extraction.PdfFeatures.PdfSegment import PdfSegment
from metadata_extraction.PdfFeatures.PdfTag import PdfTag
from metadata_extraction.PdfFeatures.Rectangle import Rectangle
from metadata_extraction.PdfFeatures.embeddings_models import (
    multilingual_sentence_embeddings_model,
    sentence_embeddings_model,
)
from metadata_extraction.XmlFile import XmlFile
from segment_selector.LightGBM_30Features_OneHotOneLetter.predict import set_tag_types


class PdfFeatures:
    def __init__(self, pages: List[PdfPage], fonts: List[PdfFont], file_name="", file_type: str = ""):
        self.pages = pages
        self.fonts = fonts
        self.file_name = file_name
        self.file_type = file_type
        self.pdf_path = ""
        self.pdf_segments: List[PdfSegment] = list()

    @staticmethod
    def from_pdfalto(file_path) -> "PdfFeatures":
        with open(file_path, mode="r") as xml_file:
            xml_content = BeautifulSoup(xml_file.read(), "lxml-xml")

        return PdfFeatures.from_xml_content(xml_content)

    @staticmethod
    def from_labeled_data(labeled_data_path) -> Optional["PdfFeatures"]:
        if not exists(os.path.join(labeled_data_path, PdfAnnotation.LABELS_FILE_NAME)):
            return None

        file_name = os.path.basename(labeled_data_path)
        extraction_name = Path(labeled_data_path).parts[-2]
        with open(os.path.join(labeled_data_path, f"{file_name}.xml"), mode="r") as xml_file:
            xml_content = BeautifulSoup(xml_file.read(), "lxml-xml")

        pdf_features = PdfFeatures.from_xml_content(xml_content, file_name=file_name)

        with open(os.path.join(labeled_data_path, PdfAnnotation.LABELS_FILE_NAME)) as f:
            all_annotations = json.load(f)

        annotations_dict = [a for a in all_annotations["annotations"] if a["label"]["text"].lower() == extraction_name]
        annotations = [
            PdfAnnotation(page_number=annotation_dict["page"] + 1, bounds=Rectangle(**annotation_dict["bounds"]))
            for annotation_dict in annotations_dict
        ]
        segmentation_json = os.path.join(labeled_data_path, f"{file_name}.json")
        with open(segmentation_json, mode="r") as json_file:
            pdf_features.set_segments_from_dict(json.load(json_file), annotations)

        return pdf_features

    @staticmethod
    def from_xml_content(xml_content, file_name=""):
        pages = list()
        fonts = [PdfFont.from_text_style_pdfalto(xml_tag) for xml_tag in xml_content.find_all("TextStyle")]

        for xml_page in xml_content.find_all("Page"):
            pages.append(PdfPage.from_pdfalto(xml_page, fonts))

        return PdfFeatures(pages, fonts, file_name=file_name)

    @staticmethod
    def get_tags_from_pdf_features(pdf_features: "PdfFeatures") -> List[PdfTag]:
        pdf_tags: List[PdfTag] = list()
        for pdf_tag in pdf_features.get_tags():
            pdf_tags.append(pdf_tag)

        return pdf_tags

    def get_tags(self) -> List[PdfTag]:
        tags: List[PdfTag] = list()
        for page in self.pages:
            for tag in page.tags:
                if tag.content.strip() == "":
                    continue
                tags.append(tag)
        return tags

    def set_segments_from_dict(self, segmentation_dict: Dict[str, str], annotations: List[PdfAnnotation]):
        for index, segment_dict in enumerate(segmentation_dict["paragraphs"]):
            segment = PdfSegment.from_segment_dict(index, segment_dict)
            segment.set_ml_label_from_annotations(annotations)
            self.pdf_segments.append(segment)

    def set_segments_from_segmentation_data(self, segmentation_data: SegmentationData):
        pdf_segments_to_merge = defaultdict(list)
        pdf_segments_from_segmentation = [
            PdfSegment.from_segment_box(segment_box) for segment_box in segmentation_data.xml_segments_boxes
        ]
        for tag in self.get_tags():
            segment_from_tag = PdfSegment.from_pdf_tag(tag)

            intersects_segmentation = [
                segment for segment in pdf_segments_from_segmentation if segment.intersects(segment_from_tag)
            ]

            if intersects_segmentation:
                pdf_segments_to_merge[intersects_segmentation[0]].append(segment_from_tag)
            else:
                self.pdf_segments.append(segment_from_tag)

        self.pdf_segments.extend(
            [
                PdfSegment.from_list_to_merge(each_pdf_segments_to_merge)
                for each_pdf_segments_to_merge in pdf_segments_to_merge.values()
            ]
        )
        self.pdf_segments.sort(key=lambda x: (x.page_number, x.bounding_box.top, x.bounding_box.left))

    def set_ml_label_from_segmentation_data(self, segmentation_data):
        for label_segment_box in segmentation_data.label_segments_boxes:
            label_boundaries = Rectangle.from_segment_box(label_segment_box)
            for segment in self.pdf_segments:
                if segment.page_number != label_segment_box.page_number:
                    continue
                if segment.is_selected(label_boundaries):
                    segment.ml_label = 1

    @staticmethod
    def get_blank():
        return PdfFeatures([], [], "", "")

    @staticmethod
    def set_embeddings(pdf_features: "PdfFeatures") -> "PdfFeatures":
        text_contents = [pdf_segment.text_content for pdf_segment in pdf_features.pdf_segments]

        if not text_contents:
            return pdf_features

        sentence_embeddings = sentence_embeddings_model(text_contents)
        embeddings_list = [list(x.numpy()) for x in sentence_embeddings]

        sentence_embeddings = multilingual_sentence_embeddings_model(text_contents)
        multilingual_embeddings_list = [list(x.numpy()) for x in sentence_embeddings]
        for index, pdf_segment in enumerate(pdf_features.pdf_segments):
            pdf_segment.embeddings = embeddings_list[index]
            pdf_segment.multilingual_embeddings = multilingual_embeddings_list[index]

        return pdf_features

    @staticmethod
    def from_xml_file(xml_file: XmlFile, segmentation_data: SegmentationData) -> Optional["PdfFeatures"]:
        try:
            xml_file_content = pathlib.Path(xml_file.xml_file_path).read_bytes()
        except FileNotFoundError:
            return PdfFeatures.get_blank()

        xml = BeautifulSoup(xml_file_content, "lxml-xml")
        pdf_features = PdfFeatures.from_xml_content(xml, file_name=xml_file.xml_file)
        segmentation_data.rescale(page_width=pdf_features.pages[0].page_width, page_height=pdf_features.pages[0].page_height)
        pdf_features.set_segments_from_segmentation_data(segmentation_data)
        pdf_features.set_ml_label_from_segmentation_data(segmentation_data)
        set_tag_types(pdf_features)
        pdf_features = PdfFeatures.set_embeddings(pdf_features)
        return pdf_features
