from segments_boxes.SegmentsBoxes import SegmentsBoxes


class SegmentedXml:
    def __init__(self, xml_file: bytes, segments_boxes: SegmentsBoxes):
        self.segments_boxes = segments_boxes
        self.xml_file = xml_file

