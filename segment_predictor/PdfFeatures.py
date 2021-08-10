import hashlib
from typing import List
from statistics import mode
from segment_predictor.SegmentTag import SegmentTag


class PdfFeatures:
    def __init__(self, segment_tags: List[SegmentTag]):
        self.page_height = float(segment_tags[0].page_height)
        self.page_width = float(segment_tags[0].page_width)
        self.fonts = list({segment_tag.font for segment_tag in segment_tags if segment_tag.font})
        self.segment_tags = segment_tags
        self.lines_space_mode: float = 0
        self.right_space_mode: float = 0
        self.left_space_mode: float = 0
        self.font_size_mode: float = 0
        self.font_family_mode: int = 0
        self.font_family_mode_normalized: float = 0
        self.get_modes()

    def get_modes(self):
        line_spaces, right_spaces, left_spaces = [0], [0], [0]

        for segment_tag in self.segment_tags:
            right_spaces.append(self.page_width - segment_tag.right)
            left_spaces.append(segment_tag.left)
            bottom_tags_tops = [bottom_tag.top for bottom_tag in self.segment_tags if
                                bottom_tag.page_number == segment_tag.page_number and bottom_tag.top > segment_tag.bottom]

            if bottom_tags_tops:
                line_spaces.append(max(bottom_tags_tops) - segment_tag.bottom)

        self.lines_space_mode = mode(line_spaces)
        self.left_space_mode = mode(left_spaces)
        self.right_space_mode = mode(right_spaces)
        self.font_size_mode = mode([segment_tag.font.size for segment_tag in self.segment_tags if segment_tag.font])
        font_family_name_mode = mode([segment_tag.font.family for segment_tag in self.segment_tags if segment_tag.font])
        self.font_family_mode = abs(int(str(hashlib.sha256(font_family_name_mode.encode('utf-8')).hexdigest())[:8], 16))
        self.font_family_mode_normalized = float(f'{str(self.font_family_mode)[0]}.{str(self.font_family_mode)[1:]}')
