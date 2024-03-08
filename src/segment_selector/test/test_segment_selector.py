import json
import shutil
from os import makedirs
from os.path import exists, join
from time import time
from unittest import TestCase

from config import APP_PATH, DATA_PATH
from data.LabeledData import LabeledData
from data.SegmentBox import SegmentBox
from data.SegmentationData import SegmentationData
from metadata_extraction.PdfData import PdfData
from metadata_extraction.XmlFile import XmlFile
from segment_selector.SegmentSelector import SegmentSelector
from pdf_token_type_labels.TokenType import TokenType


class TestSegmentSelector(TestCase):
    TENANT = "segment_selector_test"
    extraction_id = "extraction_id"
    TEST_XML_NAME = "test.xml"

    TEST_XML_PATH = join(APP_PATH, "tenant_test", extraction_id, "xml_to_train", TEST_XML_NAME)
    BASE_PATH = join(DATA_PATH, TENANT, extraction_id)

    labels = SegmentBox(left=400, top=115, width=74, height=9, page_number=1, type=TokenType.TITLE)
    LABELED_DATA_JSON = {
        "tenant": TENANT,
        "id": extraction_id,
        "xml_file_name": TEST_XML_NAME,
        "language_iso": "en",
        "label_text": "text",
        "page_width": 612,
        "page_height": 792,
        "xml_segments_boxes": [],
        "label_segments_boxes": [json.loads(labels.model_dump_json())],
    }

    XML_FILE = XmlFile(
        tenant=TENANT,
        extraction_id=extraction_id,
        to_train=True,
        xml_file_name=TEST_XML_NAME,
    )

    def setUp(self):
        shutil.rmtree(join(DATA_PATH, TestSegmentSelector.TENANT), ignore_errors=True)

        makedirs(join(TestSegmentSelector.BASE_PATH, "xml_to_train"))
        test_folder_path = join(TestSegmentSelector.BASE_PATH, "xml_to_train", TestSegmentSelector.TEST_XML_NAME)
        shutil.copy(self.TEST_XML_PATH, test_folder_path)

    def tearDown(self):
        shutil.rmtree(join(DATA_PATH, TestSegmentSelector.TENANT), ignore_errors=True)

    def test_create_model(self):
        segmentation_data = SegmentationData.from_labeled_data(LabeledData(**TestSegmentSelector.LABELED_DATA_JSON))
        pdf_segments = PdfData.from_xml_file(TestSegmentSelector.XML_FILE, segmentation_data, [])

        segment_selector = SegmentSelector(TestSegmentSelector.TENANT, TestSegmentSelector.extraction_id)
        model_created, error = segment_selector.create_model(pdfs_metadata=[pdf_segments])

        self.assertTrue(model_created)
        self.assertEqual("", error)
        self.assertTrue(exists(join(TestSegmentSelector.BASE_PATH, "segment_predictor_model", "model.model")))
        self.assertFalse(exists(join(TestSegmentSelector.BASE_PATH, "multilingual_segment_predictor_model", "model.model")))

    def test_create_model_load_test(self):
        start = time()
        segmentation_data = SegmentationData.from_labeled_data(LabeledData(**TestSegmentSelector.LABELED_DATA_JSON))
        for i in range(20):
            PdfData.from_xml_file(TestSegmentSelector.XML_FILE, segmentation_data, [])

        print(time() - start, "create model")

    def test_set_extraction_segments(self):
        segmentation_data = SegmentationData.from_labeled_data(LabeledData(**TestSegmentSelector.LABELED_DATA_JSON))

        pdf_features = PdfData.from_xml_file(TestSegmentSelector.XML_FILE, segmentation_data, [])
        segment_selector = SegmentSelector(TestSegmentSelector.TENANT, TestSegmentSelector.extraction_id)
        segment_selector.create_model(pdfs_metadata=[pdf_features])

        segment_selector = SegmentSelector(TestSegmentSelector.TENANT, TestSegmentSelector.extraction_id)
        segment_selector.set_extraction_segments(pdfs_metadata=[pdf_features])

        extraction_segments = [x for x in pdf_features.pdf_metadata_segments if x.ml_label]
        self.assertEqual(1, len(extraction_segments))
        self.assertEqual(1, extraction_segments[0].page_number)
        self.assertEqual("Original: English", extraction_segments[0].text_content)
