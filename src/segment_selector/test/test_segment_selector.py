import shutil
from os import makedirs
from os.path import exists, join
from pathlib import Path
from unittest import TestCase

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

    DOCKER_VOLUME_PATH = join(Path(__file__).parent, "..", "..", "..", "docker_volume")

    TEST_XML_PATH = join(DOCKER_VOLUME_PATH, "tenant_test", PROPERTY_NAME, "xml_to_train", TEST_XML_NAME)
    BASE_PATH = join(DOCKER_VOLUME_PATH, TENANT, PROPERTY_NAME)

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
        shutil.rmtree(join(TestSegmentSelector.DOCKER_VOLUME_PATH, TestSegmentSelector.TENANT), ignore_errors=True)

        makedirs(join(TestSegmentSelector.BASE_PATH, "xml_to_train"))
        test_folder_path = join(TestSegmentSelector.BASE_PATH, "xml_to_train", TestSegmentSelector.TEST_XML_NAME)
        shutil.copy(self.TEST_XML_PATH, test_folder_path)

    def tearDown(self):
        shutil.rmtree(join(TestSegmentSelector.DOCKER_VOLUME_PATH, TestSegmentSelector.TENANT), ignore_errors=True)

    def test_create_model(self):
        segmentation_data = SegmentationData.from_labeled_data(LabeledData(**TestSegmentSelector.LABELED_DATA_JSON))

        pdf_features = PdfFeatures.from_xml_file(TestSegmentSelector.XML_FILE, segmentation_data)
        segment_selector = SegmentSelector(TestSegmentSelector.TENANT, TestSegmentSelector.PROPERTY_NAME)
        model_created, error = segment_selector.create_model(pdfs_features=[pdf_features], multilingual=False)

        self.assertTrue(model_created)
        self.assertEqual("", error)
        self.assertTrue(exists(join(TestSegmentSelector.BASE_PATH, "segment_predictor_model", "model.model")))
        self.assertFalse(exists(join(TestSegmentSelector.BASE_PATH, "multilingual_segment_predictor_model", "model.model")))

    def test_create_model_multilingual(self):
        segmentation_data = SegmentationData.from_labeled_data(LabeledData(**TestSegmentSelector.LABELED_DATA_JSON))

        pdf_features = PdfFeatures.from_xml_file(TestSegmentSelector.XML_FILE, segmentation_data)
        segment_selector = SegmentSelector(TestSegmentSelector.TENANT, TestSegmentSelector.PROPERTY_NAME)
        model_created, error = segment_selector.create_model(pdfs_features=[pdf_features], multilingual=True)

        self.assertTrue(model_created)
        self.assertEqual("", error)
        self.assertFalse(exists(join(TestSegmentSelector.BASE_PATH, "segment_predictor_model", "model.model")))
        self.assertTrue(exists(join(TestSegmentSelector.BASE_PATH, "multilingual_segment_predictor_model", "model.model")))

    def test_set_extraction_segments(self):
        segmentation_data = SegmentationData.from_labeled_data(LabeledData(**TestSegmentSelector.LABELED_DATA_JSON))

        pdf_features = PdfFeatures.from_xml_file(TestSegmentSelector.XML_FILE, segmentation_data)
        segment_selector = SegmentSelector(TestSegmentSelector.TENANT, TestSegmentSelector.PROPERTY_NAME)
        segment_selector.create_model(pdfs_features=[pdf_features], multilingual=False)

        segment_selector = SegmentSelector(TestSegmentSelector.TENANT, TestSegmentSelector.PROPERTY_NAME)
        segment_selector.set_extraction_segments(pdfs_features=[pdf_features])

        extraction_segments = [x for x in pdf_features.pdf_segments if x.ml_label]
        self.assertEqual(1, len(extraction_segments))
        self.assertEqual(1, extraction_segments[0].page_number)
        self.assertEqual("Original: English", extraction_segments[0].text_content)

    def test_set_extraction_segments_multilingual(self):
        segmentation_data = SegmentationData.from_labeled_data(LabeledData(**TestSegmentSelector.LABELED_DATA_JSON))

        pdf_features = PdfFeatures.from_xml_file(TestSegmentSelector.XML_FILE, segmentation_data)
        segment_selector = SegmentSelector(TestSegmentSelector.TENANT, TestSegmentSelector.PROPERTY_NAME)
        segment_selector.create_model(pdfs_features=[pdf_features], multilingual=True)

        segment_selector = SegmentSelector(TestSegmentSelector.TENANT, TestSegmentSelector.PROPERTY_NAME)
        segment_selector.set_extraction_segments(pdfs_features=[pdf_features])

        extraction_segments = [x for x in pdf_features.pdf_segments if x.ml_label]
        self.assertEqual(1, len(extraction_segments))
        self.assertEqual(1, extraction_segments[0].page_number)
        self.assertEqual("Original: English", extraction_segments[0].text_content)

    def test_remove_multilingual_model_when_new_model(self):
        segmentation_data = SegmentationData.from_labeled_data(LabeledData(**TestSegmentSelector.LABELED_DATA_JSON))

        pdf_features = PdfFeatures.from_xml_file(TestSegmentSelector.XML_FILE, segmentation_data)
        segment_selector = SegmentSelector(TestSegmentSelector.TENANT, TestSegmentSelector.PROPERTY_NAME)

        multilingual_model = join(TestSegmentSelector.BASE_PATH, "multilingual_segment_predictor_model", "model.model")
        makedirs(Path(multilingual_model).parent)
        shutil.copy(TestSegmentSelector.TEST_XML_PATH, multilingual_model)

        segment_selector.create_model(pdfs_features=[pdf_features], multilingual=False)

        self.assertTrue(exists(join(TestSegmentSelector.BASE_PATH, "segment_predictor_model", "model.model")))
        self.assertFalse(exists(multilingual_model))

    def test_remove_english_model_when_new_model(self):
        segmentation_data = SegmentationData.from_labeled_data(LabeledData(**TestSegmentSelector.LABELED_DATA_JSON))

        pdf_features = PdfFeatures.from_xml_file(TestSegmentSelector.XML_FILE, segmentation_data)
        segment_selector = SegmentSelector(TestSegmentSelector.TENANT, TestSegmentSelector.PROPERTY_NAME)

        english_model = join(TestSegmentSelector.BASE_PATH, "segment_predictor_model", "model.model")
        makedirs(Path(english_model).parent)
        shutil.copy(TestSegmentSelector.TEST_XML_PATH, english_model)

        segment_selector.create_model(pdfs_features=[pdf_features], multilingual=True)

        self.assertTrue(exists(join(TestSegmentSelector.BASE_PATH, "multilingual_segment_predictor_model", "model.model")))
        self.assertFalse(exists(english_model))
