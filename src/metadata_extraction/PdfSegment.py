from statistics import mode

from pdf_features.Rectangle import Rectangle
from data.SegmentBox import SegmentBox
from pdf_token_type_labels.TokenType import TokenType
from pdf_features.PdfToken import PdfToken


class PdfSegment:
    def __init__(
        self, page_number: int, bounding_box: Rectangle, text_content: str, segment_type: TokenType = TokenType.TEXT
    ):
        self.page_number = page_number
        self.bounding_box = bounding_box
        self.text_content = text_content
        self.ml_label = 0
        self.segment_type: TokenType = segment_type

    def is_selected(self, bounding_box: Rectangle):
        if bounding_box.bottom < self.bounding_box.top or self.bounding_box.bottom < bounding_box.top:
            return False

        if bounding_box.right < self.bounding_box.left or self.bounding_box.right < bounding_box.left:
            return False

        return True

    def intersects(self, pdf_segment: "PdfSegment"):
        if self.page_number != pdf_segment.page_number:
            return False

        return pdf_segment.bounding_box.get_intersection_percentage(self.bounding_box) > 50

    @staticmethod
    def from_segment_dict(index, segment_dict) -> "PdfSegment":
        return PdfSegment(
            page_number=segment_dict["page_number"],
            segment_ids=index,
            bounding_box=Rectangle.from_segment_dict(segment_dict),
            text_content=segment_dict["text"],
        )

    @staticmethod
    def from_segment_box(segment_box: SegmentBox) -> "PdfSegment":
        return PdfSegment(
            page_number=segment_box.page_number,
            bounding_box=segment_box.get_bounding_box(),
            text_content="",
            segment_type=segment_box.segment_type,
        )

    @staticmethod
    def from_pdf_token(pdf_token: PdfToken):
        return PdfSegment(
            page_number=pdf_token.page_number,
            bounding_box=pdf_token.bounding_box,
            text_content=pdf_token.content,
            segment_type=pdf_token.token_type,
        )

    @staticmethod
    def from_list_to_merge(pdf_segments_to_merge: list["PdfSegment"]):
        text_content = " ".join([pdf_segment.text_content for pdf_segment in pdf_segments_to_merge])
        bounding_box = Rectangle.merge_rectangles([pdf_segment.bounding_box for pdf_segment in pdf_segments_to_merge])
        segment_type = mode([segment.segment_type for segment in pdf_segments_to_merge])
        return PdfSegment(
            page_number=pdf_segments_to_merge[0].page_number,
            bounding_box=bounding_box,
            text_content=text_content,
            segment_type=segment_type,
        )

    def get_segment_box(self):
        return SegmentBox(
            left=self.bounding_box.left,
            top=self.bounding_box.top,
            width=self.bounding_box.width,
            height=self.bounding_box.height,
            page_number=self.page_number,
        )