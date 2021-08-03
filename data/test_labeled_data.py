from unittest import TestCase

import mongomock as mongomock
import pymongo as pymongo

from data.ExtractionId import ExtractionId
from data.Label import Label
from data.LabeledData import LabeledData
from data.SegmentedXml import SegmentedXml
from segments_boxes.SegmentsBoxes import SegmentsBoxes, SegmentBox


class TestLabeledData(TestCase):
    @mongomock.patch(servers=['mongodb://mongo:27017'])
    def test_save(self):
        client = pymongo.MongoClient('mongodb://mongo:27017')

        label_segments_boxes = SegmentsBoxes(page_width=1, page_height=1, segment_boxes=[SegmentBox(1, 2, 3, 4, 5)])
        label = Label(text='text', segments_boxes=label_segments_boxes)
        extraction_id = ExtractionId(extraction_id='extraction_id', tenant='tenant')
        xml_segments_boxes = SegmentsBoxes(page_width=2, page_height=2, segment_boxes=[SegmentBox(6, 7, 8, 9, 10)])
        segmented_xml = SegmentedXml(xml_file=b'xml_file', segments_boxes=xml_segments_boxes)
        labeled_data = LabeledData(label_id='label_id', extraction_id=extraction_id, segmented_xml=segmented_xml,
                                   label=label)

        labeled_data.save()

        labeled_data_document = client.pdf_information_extraction.labeleddata.find_one()

        self.assertEqual('label_id', labeled_data_document['label_id'])
        self.assertEqual('extraction_id', labeled_data_document['extraction_id'])
        self.assertEqual('tenant', labeled_data_document['tenant'])
        self.assertEqual(
            '{"pageWidth": 2, "pageHeight": 2, "segments": [{"pageNumber": 6, "left": 7, "top": 8, "width": 9, '
            '"height": 10}]}',
            labeled_data_document['xml_segments_boxes'])
        self.assertEqual('text', labeled_data_document['label_text'])
        self.assertEqual(
            '{"pageWidth": 1, "pageHeight": 1, "segments": [{"pageNumber": 1, "left": 2, "top": 3, "width": 4, '
            '"height": 5}]}',
            labeled_data_document['label_segments_boxes'])

    @mongomock.patch(servers=['mongodb://mongo:27017'])
    def test_save_no_tenant(self):
        client = pymongo.MongoClient('mongodb://mongo:27017')

        label_segments_boxes = SegmentsBoxes(page_width=1, page_height=1, segment_boxes=[SegmentBox(1, 2, 3, 4, 5)])
        label = Label(text='no_tenant', segments_boxes=label_segments_boxes)
        extraction_id = ExtractionId(extraction_id='no_tenant_id')
        xml_segments_boxes = SegmentsBoxes(page_width=2, page_height=2, segment_boxes=[SegmentBox(6, 7, 8, 9, 10)])
        segmented_xml = SegmentedXml(xml_file=b'xml_file', segments_boxes=xml_segments_boxes)
        labeled_data = LabeledData(label_id='no_tenant_label_id', extraction_id=extraction_id, segmented_xml=segmented_xml,
                                   label=label)

        labeled_data.save()

        labeled_data_document = client.pdf_information_extraction.labeleddata.find_one()

        self.assertEqual('no_tenant_label_id', labeled_data_document['label_id'])
        self.assertEqual('no_tenant_id', labeled_data_document['extraction_id'])
        self.assertNotIn('tenant', labeled_data_document)
        self.assertEqual(
            '{"pageWidth": 2, "pageHeight": 2, "segments": [{"pageNumber": 6, "left": 7, "top": 8, "width": 9, '
            '"height": 10}]}',
            labeled_data_document['xml_segments_boxes'])
        self.assertEqual('no_tenant', labeled_data_document['label_text'])
        self.assertEqual(
            '{"pageWidth": 1, "pageHeight": 1, "segments": [{"pageNumber": 1, "left": 2, "top": 3, "width": 4, '
            '"height": 5}]}',
            labeled_data_document['label_segments_boxes'])

