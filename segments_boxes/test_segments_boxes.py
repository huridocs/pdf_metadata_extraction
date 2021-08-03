from unittest import TestCase
from segments_boxes.SegmentsBoxes import SegmentsBoxes, SegmentBox


class TestSegmentsBoxes(TestCase):
    def test_to_json(self):
        segments_boxes = SegmentsBoxes(1, 2, [SegmentBox(1, 1, 2, 3, 4), SegmentBox(2, 0.1, 0.2, 0.3, 0.4)])
        json_result = '{"pageWidth": 1, "pageHeight": 2, "segments": [' \
                      '{"pageNumber": 1, "left": 1, "top": 2, "width": 3, "height": 4}, ' \
                      '{"pageNumber": 2, "left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4}]}'
        self.assertEqual(segments_boxes.to_json(), json_result)
