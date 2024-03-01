from typing import Optional

from paragraph_extraction_trainer.Paragraph import Paragraph

from data.SegmentationData import SegmentationData
from pdf_features.PdfFeatures import PdfFeatures

from metadata_extraction.FilterValidSegmentsPages import FilterValidSegmentPages
from metadata_extraction.PdfMetadataSegment import PdfMetadataSegment
from metadata_extraction.XmlFile import XmlFile


class PdfMetadata:
    def __init__(self, pdf_features: Optional[PdfFeatures], file_name="", file_type: str = ""):
        self.pdf_features: PdfFeatures = pdf_features
        self.file_name = file_name
        self.file_type = file_type
        self.pdf_path = ""
        self.pdf_metadata_segments: list[PdfMetadataSegment] = list()

    def set_segments_from_paragraphs(self, paragraphs: list[Paragraph]):
        for paragraph in paragraphs:
            self.pdf_metadata_segments.append(PdfMetadataSegment.from_pdf_tokens(paragraph.tokens))
        self.pdf_metadata_segments.sort(key=lambda x: (x.page_number, x.bounding_box.top, x.bounding_box.left))

    def set_segments_from_segmentation_data(self, segmentation_data: SegmentationData):
        pdf_segments_to_merge = dict()
        pdf_segments_from_segmentation = [
            segment_box.to_pdf_segment() for segment_box in segmentation_data.xml_segments_boxes
        ]
        for page, token in self.pdf_features.loop_tokens():
            segment_from_tag: PdfMetadataSegment = PdfMetadataSegment.from_pdf_token(token)

            intersects_segmentation = [
                segmentation_segment
                for segmentation_segment in pdf_segments_from_segmentation
                if segmentation_segment.intersects(segment_from_tag)
            ]

            if not intersects_segmentation:
                self.pdf_metadata_segments.append(segment_from_tag)
                continue

            segment_from_tag.segment_type = intersects_segmentation[0].segment_type
            pdf_segments_to_merge.setdefault(intersects_segmentation[0], []).append(segment_from_tag)

        self.pdf_metadata_segments.extend(
            [
                PdfMetadataSegment.from_list_to_merge(each_pdf_segments_to_merge)
                for each_pdf_segments_to_merge in pdf_segments_to_merge.values()
            ]
        )
        self.pdf_metadata_segments.sort(key=lambda x: (x.page_number, x.bounding_box.top, x.bounding_box.left))

    def set_ml_label_from_segmentation_data(self, segmentation_data: SegmentationData):
        for label_segment_box in segmentation_data.label_segments_boxes:
            for segment in self.pdf_metadata_segments:
                if segment.page_number != label_segment_box.page_number:
                    continue
                if segment.is_selected(label_segment_box.get_bounding_box()):
                    segment.ml_label = 1

    @staticmethod
    def get_blank():
        return PdfMetadata(None)

    @staticmethod
    def from_xml_file(xml_file: XmlFile, segmentation_data: SegmentationData, pages_numbers: list[int]) -> "PdfMetadata":
        try:
            file_content: str = open(xml_file.xml_file_path).read()
        except FileNotFoundError:
            return PdfMetadata.get_blank()

        xml_file_content = FilterValidSegmentPages.filter_xml_pages(file_content, pages_numbers)
        pdf_features = PdfFeatures.from_poppler_etree_content(xml_file.xml_file_path, xml_file_content)

        if not pdf_features:
            return PdfMetadata.get_blank()

        pdf_segments = PdfMetadata(pdf_features)
        pdf_segments.set_segments_from_segmentation_data(segmentation_data)
        pdf_segments.set_ml_label_from_segmentation_data(segmentation_data)

        return pdf_segments
