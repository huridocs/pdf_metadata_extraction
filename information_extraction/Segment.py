import re
from typing import List
import numpy as np

from data.SegmentBox import SegmentBox
from information_extraction.PdfFeatures import PdfFeatures
from information_extraction.SegmentTag import SegmentTag


class Segment(object):
    def __init__(self, segment_tag: SegmentTag, pdf_features: PdfFeatures):
        self.ml_class_label: int = 0
        self.confidence: float = 0
        self.page_number = segment_tag.page_number
        self.segment_tags = [segment_tag]
        self.pdf_features = pdf_features
        self.page_width = self.pdf_features.page_width
        self.page_height = self.pdf_features.page_height
        self.text_content: str = ''
        self.text_len: int = 0
        self.top: float = 0
        self.left: float = 0
        self.right: float = 0
        self.bottom: float = 0
        self.height: float = 0
        self.width: float = 0
        self.font_size: float = 0.0
        self.font_family: str = ''
        self.font_color: str = ''
        self.line_height: int = 0
        self.numbers_quantity: int = 0
        self.numbers_percentage: float = 0
        self.starts_upper: bool = False
        self.starts_number: bool = False
        self.starts_number_bar: bool = False
        self.starts_letter_dot: bool = False
        self.starts_with_square_brackets: bool = False
        self.starts_with_roman_numbers: bool = False
        self.uppercase: bool = False
        self.last_tag: SegmentTag = None
        self.bold: float = False
        self.bold_tag_number: int = 0
        self.italics: float = False
        self.italics_tag_number: int = 0
        self.first_word_type: int = 100
        self.second_word_type: int = 100
        self.third_word_type: int = 100
        self.fourth_word_type: int = 100
        self.last_word_type: int = 100
        self.dots_percentage: float = 0
        self.set_features()

    def initialize_features(self):
        self.page_width = self.pdf_features.page_width
        self.page_height = self.pdf_features.page_height
        self.ml_class_label: int = 0
        self.text_content: str = ''
        self.text_len: int = 0
        self.top: float = 0
        self.left: float = 0
        self.right: float = 0
        self.bottom: float = 0
        self.height: float = 0
        self.width: float = 0
        self.font_size: float = 0.0
        self.font_family: str = ''
        self.font_color: str = ''
        self.line_height: int = 0
        self.numbers_quantity: int = 0
        self.numbers_percentage: float = 0
        self.starts_upper: bool = False
        self.starts_number: bool = False
        self.starts_number_bar: bool = False
        self.starts_letter_dot: bool = False
        self.starts_with_square_brackets: bool = False
        self.starts_with_roman_numbers: bool = False
        self.uppercase: bool = False
        self.last_tag: SegmentTag = None
        self.bold: float = False
        self.bold_tag_number: int = 0
        self.italics: float = False
        self.italics_tag_number: int = 0
        self.first_word_type: int = 100
        self.second_word_type: int = 100
        self.third_word_type: int = 100
        self.fourth_word_type: int = 100
        self.last_word_type: int = 100
        self.dots_percentage: float = 0

    def set_features(self):
        self.initialize_features()
        self.font_family = self.segment_tags[0].font.family
        self.font_color = self.segment_tags[0].font.color
        self.line_height = self.segment_tags[0].height
        self.top = self.segment_tags[0].top
        self.left = self.segment_tags[0].left
        self.right = self.segment_tags[0].left + self.segment_tags[0].width
        self.bottom = self.segment_tags[0].top + self.segment_tags[0].height
        words: List[str] = list()

        font_sizes = list()
        for tag in self.segment_tags:
            words.extend(tag.text.split())
            self.top = min(self.top, tag.top)
            self.left = min(self.left, tag.left)
            self.right = max(self.right, tag.left + tag.width)
            self.bottom = max(self.bottom, tag.top + tag.height)
            self.bold_tag_number = self.bold_tag_number + 1 if tag.font.bold else self.bold_tag_number
            self.italics_tag_number = self.italics_tag_number + 1 if tag.font.italics else self.italics_tag_number
            font_sizes.append(tag.font.size)
            if self.tag_after_last_tag(tag):
                self.last_tag = tag

        self.top = self.top / self.page_height
        self.bottom = self.bottom / self.page_height
        self.right = self.right / self.page_width
        self.left = self.left / self.page_width

        self.text_content = ' '.join(words)
        self.text_len = len(self.text_content)
        self.dots_percentage = self.text_content.count('.') / self.text_len if self.text_len > 0 else 0
        self.height = self.bottom - self.top
        self.width = self.right - self.left
        self.font_size = np.mean(font_sizes)
        self.numbers_quantity = len(list(filter(lambda x: x.isdigit(), self.text_content)))
        self.numbers_percentage = self.numbers_quantity / self.text_len if self.text_len > 0 else 0
        self.bold = self.bold_tag_number / len(self.segment_tags)
        self.italics = self.italics_tag_number / len(self.segment_tags)
        self.starts_upper = self.text_content[0].isupper()
        self.starts_number = self.text_content[0].isdigit()
        self.starts_number_bar = len(re.findall(r"^[0-9]\/", self.text_content)) == 1
        self.starts_letter_dot = len(re.findall(r"^[a-zA-Z]\.", self.text_content)) == 1
        self.uppercase = self.text_content.upper() == self.text_content
        first_characters = self.text_content.split(' ')[0].split('.')[0]
        roman_numbers = ''.join(filter(lambda x: x in 'IVXLCDM', first_characters))
        if len(roman_numbers) > 0 and roman_numbers == first_characters:
            self.starts_with_roman_numbers = True
        self.starts_with_square_brackets = self.text_content[0] == '['

    def get_features_array(self) -> np.array:
        return np.array([
            self.page_number,
            self.pdf_features.font_size_mode,
            self.pdf_features.lines_space_mode,
            self.pdf_features.font_family_mode_normalized,
            self.pdf_features.page_width / 5000,
            self.pdf_features.page_height / 5000,
            self.pdf_features.left_space_mode / self.page_width,
            self.bold,
            self.italics,
            self.text_len,
            self.top,
            self.bottom,
            self.height,
            self.width,
            self.font_size / self.pdf_features.font_size_mode,
            self.line_height,
            self.numbers_percentage,
            1 if self.starts_upper else 0,
            1 if self.starts_number else 0,
            self.starts_number_bar,
            self.numbers_quantity,
            self.starts_with_square_brackets,
            self.starts_letter_dot,
            self.dots_percentage,
            1 if self.uppercase else 0
        ])

    def intersects_with_box(self, page_width: float, page_height: float, segment_box: SegmentBox):
        if segment_box.page_number != self.page_number:
            return False

        box_top = segment_box.top / page_height
        box_bottom = (segment_box.top + segment_box.height) / page_height

        if box_bottom < self.top or self.bottom < box_top:
            return False

        box_left = segment_box.left / page_width
        box_right = (segment_box.left + segment_box.width) / page_width

        if box_right < self.left or self.right < box_left:
            return False

        return True

    def intersects_with_boxes(self, page_width: float, page_height: float, segment_boxes: List[SegmentBox]):
        for index, segment_box in enumerate(segment_boxes):
            if self.intersects_with_box(page_width, page_height, segment_box):
                return index

        return None

    def tag_after_last_tag(self, tag: SegmentTag):
        if self.last_tag is None:
            return True

        if self.last_tag.bottom < tag.bottom:
            return True

        if self.last_tag.left < tag.left:
            return True

        return False

    def set_ml_label(self, page_width: float, page_height: float, segment_boxes: List[SegmentBox]):
        for box in segment_boxes:
            if self.intersects_with_box(page_width, page_height, box):
                self.ml_class_label = 1
                break

    @staticmethod
    def merge(segments_to_merge: List['Segment']):
        if len(segments_to_merge) == 1:
            return segments_to_merge[0]

        tags = []
        for segment_to_merge in segments_to_merge:
            tags.extend(segment_to_merge.segment_tags)
        segment = segments_to_merge[0]
        segment.segment_tags = tags
        segment.set_features()
        return segment

