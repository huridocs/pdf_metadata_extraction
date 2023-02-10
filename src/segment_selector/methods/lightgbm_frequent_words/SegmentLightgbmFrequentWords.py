import re
from typing import List, Optional

import nltk
import numpy as np

from metadata_extraction.PdfFeatures.PdfFeatures import PdfFeatures
from metadata_extraction.PdfFeatures.PdfSegment import PdfSegment
from metadata_extraction.PdfFeatures.PdfTag import PdfTag
from metadata_extraction.PdfFeatures.TagType import TAG_TYPE_DICT
from segment_selector.methods.Modes import Modes

nltk.download("punkt")


class SegmentLightgbmFrequentWords:
    def __init__(self, segment_index: int, pdf_segment: PdfSegment, pdf_features: PdfFeatures, modes: Modes):
        self.modes = modes
        self.previous_title_segment = None
        self.previous_segment = None
        self.next_segment = None
        self.sentence_embeddings = pdf_segment.embeddings
        self.segment_index: float = segment_index
        self.confidence: float = 0
        self.page_number = pdf_segment.page_number
        self.page_index = pdf_segment.page_number - 1
        self.pdf_segment = pdf_segment
        self.segment_tags: List[PdfTag] = [
            pdf_tag
            for pdf_tag in pdf_features.get_tags()
            if self.page_number == pdf_tag.page_number and pdf_segment.is_selected(pdf_tag.bounding_box)
        ]
        self.segment_tags = self.segment_tags if self.segment_tags else [pdf_features.get_tags()[-1]]
        self.pdf_features = pdf_features
        self.page_width = self.pdf_features.pages[0].page_width
        self.page_height = self.pdf_features.pages[0].page_height
        self.text_content: str = ""
        self.text_len: int = 0
        self.top: float = 0
        self.left: float = 0
        self.right: float = 0
        self.bottom: float = 0
        self.height: float = 0
        self.width: float = 0
        self.font_size: float = 0.0
        self.font_family: str = ""
        self.font_color: str = ""
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
        self.last_tag: PdfSegment = None
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
        self.font_sizes = [x.font.font_size for x in self.pdf_features.get_tags()]
        self.page_width = self.pdf_features.pages[0].page_width
        self.page_height = self.pdf_features.pages[0].page_height
        self.text_content: str = ""
        self.text_len: int = 0
        self.top: float = 0
        self.left: float = 0
        self.right: float = 0
        self.bottom: float = 0
        self.height: float = 0
        self.width: float = 0
        self.font_size: float = 0.0
        self.font_family: str = ""
        self.font_color: str = ""
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
        self.last_tag: PdfSegment = None
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
        self.most_frequent_words = list()
        self.set_features()

    def set_features(self):
        self.font_family = self.segment_tags[0].font.font_id
        self.font_color = self.segment_tags[0].font.color
        self.line_height = self.segment_tags[0].font.font_size
        self.top = self.segment_tags[0].bounding_box.top
        self.left = self.segment_tags[0].bounding_box.left
        self.right = self.segment_tags[0].bounding_box.right
        self.bottom = self.segment_tags[0].bounding_box.bottom
        words: List[str] = list()

        font_sizes = list()
        for tag in self.segment_tags:
            words.extend(tag.content.split())
            self.top = min(self.top, tag.bounding_box.top)
            self.left = min(self.left, tag.bounding_box.left)
            self.right = max(self.right, tag.bounding_box.left + tag.bounding_box.width)
            self.bottom = max(self.bottom, tag.bounding_box.top + tag.bounding_box.height)
            self.bold_tag_number = self.bold_tag_number + 1 if tag.font.bold else self.bold_tag_number
            self.italics_tag_number = self.italics_tag_number + 1 if tag.font.italics else self.italics_tag_number
            font_sizes.append(tag.font.font_size)
            if self.tag_after_last_tag(tag):
                self.last_tag = tag

        self.top = self.top / self.page_height
        self.bottom = self.bottom / self.page_height
        self.right = self.right / self.page_width
        self.left = self.left / self.page_width
        self.text_content = " ".join(words)
        self.text_len = len(self.text_content)
        self.dots_percentage = self.text_content.count(".") / self.text_len if self.text_len > 0 else 0
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
        first_characters = self.text_content.split(" ")[0].split(".")[0]
        roman_numbers = "".join(filter(lambda x: x in "IVXLCDM", first_characters))
        if len(roman_numbers) > 0 and roman_numbers == first_characters:
            self.starts_with_roman_numbers = True
        self.starts_with_square_brackets = self.text_content[0] == "["

    def get_last_title_features(self):
        if not self.previous_title_segment:
            return list(np.zeros(22))

        font_size_mode = sum(self.previous_title_segment.font_sizes) / len(self.previous_title_segment.font_sizes)

        return [
            self.previous_title_segment.segment_index,
            len(self.previous_title_segment.pdf_features.pdf_segments) - self.previous_title_segment.segment_index,
            self.previous_title_segment.page_index,
            len(self.previous_title_segment.pdf_features.pages) - self.previous_title_segment.page_index,
            self.previous_title_segment.bold,
            self.previous_title_segment.italics,
            self.previous_title_segment.text_len,
            self.previous_title_segment.top,
            self.previous_title_segment.bottom,
            self.previous_title_segment.height,
            self.previous_title_segment.width,
            self.previous_title_segment.font_size / font_size_mode,
            self.previous_title_segment.line_height,
            self.previous_title_segment.numbers_percentage,
            1 if self.previous_title_segment.starts_upper else 0,
            1 if self.previous_title_segment.starts_number else 0,
            self.previous_title_segment.starts_number_bar,
            self.previous_title_segment.numbers_quantity,
            self.previous_title_segment.starts_with_square_brackets,
            self.previous_title_segment.starts_letter_dot,
            self.previous_title_segment.dots_percentage,
            1 if self.previous_title_segment.uppercase else 0,
        ]

    @staticmethod
    def get_other_segment_features(segment: "SegmentLightgbmFrequentWords"):
        if not segment:
            return list(np.zeros(22))

        font_size_mode = sum(segment.font_sizes) / len(segment.font_sizes)

        return [
            segment.segment_index,
            len(segment.pdf_features.pdf_segments) - segment.segment_index,
            segment.page_index,
            len(segment.pdf_features.pages) - segment.page_index,
            segment.bold,
            segment.italics,
            segment.text_len,
            segment.top,
            segment.bottom,
            segment.height,
            segment.width,
            segment.font_size / font_size_mode,
            segment.line_height,
            segment.numbers_percentage,
            1 if segment.starts_upper else 0,
            1 if segment.starts_number else 0,
            segment.starts_number_bar,
            segment.numbers_quantity,
            segment.starts_with_square_brackets,
            segment.starts_letter_dot,
            segment.dots_percentage,
            1 if segment.uppercase else 0,
        ]

    def get_features_array(self) -> np.array:
        font_size_average = sum(self.font_sizes) / len(self.font_sizes)

        features = np.array(
            [
                self.segment_index,
                self.page_index,
                font_size_average,
                self.modes.lines_space_mode,
                self.modes.font_family_mode_normalized,
                self.pdf_features.pages[0].page_width / 5000,
                self.pdf_features.pages[0].page_height / 5000,
                self.modes.left_space_mode / self.page_width,
                self.bold,
                self.italics,
                self.text_len,
                self.top,
                self.bottom,
                self.height,
                self.width,
                self.font_size / self.modes.font_size_mode,
                self.line_height,
                self.numbers_percentage,
                1 if self.starts_upper else 0,
                1 if self.starts_number else 0,
                self.starts_number_bar,
                self.numbers_quantity,
                self.starts_with_square_brackets,
                self.starts_letter_dot,
                self.dots_percentage,
                1 if self.uppercase else 0,
                len(self.pdf_features.pdf_segments) - self.segment_index,
                len(self.pdf_features.pages) - self.page_index,
                self.pdf_segment.segment_type,
            ]
            + self.get_other_segment_features(self.previous_segment)
            + self.get_other_segment_features(self.next_segment)
            + self.get_last_title_features()
            + self.most_frequent_words
        )
        return features

    def tag_after_last_tag(self, tag: PdfTag):
        if self.last_tag is None:
            return True

        if self.last_tag.bounding_box.bottom < tag.bounding_box.bottom:
            return True

        if self.last_tag.bounding_box.left < tag.bounding_box.left:
            return True

        return False

    @staticmethod
    def from_pdf_features(pdf_features: PdfFeatures) -> List["SegmentLightgbmFrequentWords"]:
        modes = Modes(pdf_features)
        segments: List["SegmentLightgbmFrequentWords"] = list()
        for index, pdf_segment in enumerate(pdf_features.pdf_segments):
            segment_landmarks = SegmentLightgbmFrequentWords(index, pdf_segment, pdf_features, modes)
            segments.append(segment_landmarks)

        sorted_pdf_segments = sorted(segments, key=lambda x: (x.page_index, x.top))

        previous_title_segment: Optional["SegmentLightgbmFrequentWords"] = None

        for sorted_segment in sorted_pdf_segments:
            sorted_segment.previous_title_segment = previous_title_segment
            if sorted_segment.pdf_segment.segment_type == TAG_TYPE_DICT["title"]:
                previous_title_segment = sorted_segment

        for index, sorted_segment in enumerate(sorted_pdf_segments):
            if 0 < index:
                sorted_segment.previous_segment = sorted_pdf_segments[index - 1]

            if index + 1 < len(sorted_pdf_segments):
                sorted_segment.next_segment = sorted_pdf_segments[index + 1]

        return segments

    def set_most_frequent_words(self, most_frequent_words: List[str]):
        self.most_frequent_words = [1 if w.lower() in self.text_content.lower() else 0 for w in most_frequent_words]
