import shutil
from os import makedirs
from os.path import exists, join
from time import time
from unittest import TestCase

from config import APP_PATH, DATA_PATH
from data.LabeledData import LabeledData
from data.SegmentBox import SegmentBox
from data.SegmentationData import SegmentationData
from metadata_extraction.PdfFeatures.PdfFeatures import PdfFeatures
from metadata_extraction.XmlFile import XmlFile
from segment_selector.SegmentSelector import SegmentSelector


class TestSegmentSelector(TestCase):
    TENANT = "segment_selector_test"
    PROPERTY_NAME = "property_name"
    TEST_XML_NAME = "test.xml"

    TEST_XML_PATH = join(APP_PATH, "tenant_test", PROPERTY_NAME, "xml_to_train", TEST_XML_NAME)
    BASE_PATH = join(DATA_PATH, TENANT, PROPERTY_NAME)

    LABELED_DATA_JSON = {
        "tenant": TENANT,
        "property_name": PROPERTY_NAME,
        "xml_file_name": TEST_XML_NAME,
        "language_iso": "en",
        "label_text": "text",
        "page_width": 612,
        "page_height": 792,
        "xml_segments_boxes": [],
        "label_segments_boxes": [SegmentBox(left=400, top=115, width=74, height=9, page_number=1).dict()],
    }

    XML_FILE = XmlFile(
        tenant=TENANT,
        property_name=PROPERTY_NAME,
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
        pdf_features = PdfFeatures.from_xml_file(TestSegmentSelector.XML_FILE, segmentation_data, [])

        segment_selector = SegmentSelector(TestSegmentSelector.TENANT, TestSegmentSelector.PROPERTY_NAME)
        model_created, error = segment_selector.create_model(pdfs_features=[pdf_features])

        self.assertTrue(model_created)
        self.assertEqual("", error)
        self.assertTrue(exists(join(TestSegmentSelector.BASE_PATH, "segment_predictor_model", "model.model")))
        self.assertFalse(exists(join(TestSegmentSelector.BASE_PATH, "multilingual_segment_predictor_model", "model.model")))

    def test_create_model_load_test(self):
        start = time()
        segmentation_data = SegmentationData.from_labeled_data(LabeledData(**TestSegmentSelector.LABELED_DATA_JSON))
        for i in range(20):
            PdfFeatures.from_xml_file(TestSegmentSelector.XML_FILE, segmentation_data, [])

        print(time() - start, "create model")

    def test_set_extraction_segments(self):
        segmentation_data = SegmentationData.from_labeled_data(LabeledData(**TestSegmentSelector.LABELED_DATA_JSON))

        pdf_features = PdfFeatures.from_xml_file(TestSegmentSelector.XML_FILE, segmentation_data, [])
        segment_selector = SegmentSelector(TestSegmentSelector.TENANT, TestSegmentSelector.PROPERTY_NAME)
        segment_selector.create_model(pdfs_features=[pdf_features])

        segment_selector = SegmentSelector(TestSegmentSelector.TENANT, TestSegmentSelector.PROPERTY_NAME)
        segment_selector.set_extraction_segments(pdfs_features=[pdf_features])

        extraction_segments = [x for x in pdf_features.pdf_segments if x.ml_label]
        self.assertEqual(1, len(extraction_segments))
        self.assertEqual(1, extraction_segments[0].page_number)
        self.assertEqual("Original: English", extraction_segments[0].text_content)
