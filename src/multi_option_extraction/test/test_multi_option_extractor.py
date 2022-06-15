import shutil
from os import makedirs
from os.path import join
from pathlib import Path
from unittest import TestCase

from data.Option import Option
from multi_option_extraction.MultiOptionExtractionData import MultiOptionExtractionSample, MultiOptionExtractionData
from multi_option_extraction.MultiOptionExtractor import MultiOptionExtractor


class TestMultiOptionExtractor(TestCase):
    TENANT = "multi_option_extractor_test"
    PROPERTY_NAME = "property_name"

    DOCKER_VOLUME_PATH = join(Path(__file__).parent, "..", "..", "..", "docker_volume")
    BASE_PATH = join(DOCKER_VOLUME_PATH, TENANT, PROPERTY_NAME)

    def setUp(self):
        shutil.rmtree(join(TestMultiOptionExtractor.DOCKER_VOLUME_PATH, TestMultiOptionExtractor.TENANT), ignore_errors=True)
        makedirs(TestMultiOptionExtractor.BASE_PATH)

    def tearDown(self):
        shutil.rmtree(join(TestMultiOptionExtractor.DOCKER_VOLUME_PATH, TestMultiOptionExtractor.TENANT), ignore_errors=True)

    def test_exist_model(self):
        multi_option_extractor = MultiOptionExtractor(
            tenant=TestMultiOptionExtractor.TENANT, property_name=TestMultiOptionExtractor.PROPERTY_NAME
        )
        options = [Option(id="id1", label="option 1"), Option(id="id2", label="option 2")]
        samples = [
            MultiOptionExtractionSample(text="option 1", options=[options[0]]),
            MultiOptionExtractionSample(text="option 2", options=[options[1]]),
        ]

        multi_option_extraction_data = MultiOptionExtractionData(
            multilingual=False, multi_value=False, options=options, samples=samples
        )

        multi_option_extractor.create_model(multi_option_extraction_data=multi_option_extraction_data)

        self.assertTrue(
            MultiOptionExtractor.exist_model(TestMultiOptionExtractor.TENANT, TestMultiOptionExtractor.PROPERTY_NAME)
        )
        self.assertFalse(MultiOptionExtractor.exist_model("other_tenant", TestMultiOptionExtractor.PROPERTY_NAME))
        self.assertFalse(MultiOptionExtractor.exist_model(TestMultiOptionExtractor.TENANT, "other_property_name"))

    def test_get_predictions(self):
        multi_option_extractor = MultiOptionExtractor(
            tenant=TestMultiOptionExtractor.TENANT, property_name=TestMultiOptionExtractor.PROPERTY_NAME
        )
        options = [Option(id="id1", label="option 1"), Option(id="id2", label="option 2")]
        samples = [
            MultiOptionExtractionSample(text="option 1", options=[options[0]]),
            MultiOptionExtractionSample(text="option 2", options=[options[1]]),
        ]

        multi_option_extraction_data = MultiOptionExtractionData(
            multilingual=False, multi_value=False, options=options, samples=samples
        )

        model_created, error = multi_option_extractor.create_model(multi_option_extraction_data=multi_option_extraction_data)
        predictions = multi_option_extractor.get_multi_option_predictions(["option 1", "option 2"])
        self.assertTrue(model_created)
        self.assertEqual("", error)
        self.assertTrue(2, len(predictions))
        self.assertEqual([options[0]], predictions[0].options)
        self.assertEqual([options[1]], predictions[1].options)

    def test_get_predictions_multi_value(self):
        multi_option_extractor = MultiOptionExtractor(
            tenant=TestMultiOptionExtractor.TENANT, property_name=TestMultiOptionExtractor.PROPERTY_NAME
        )
        options = [Option(id="id1", label="option 1"), Option(id="id2", label="option 2")]
        samples = [
            MultiOptionExtractionSample(text="option 1", options=[options[0]]),
            MultiOptionExtractionSample(text="option 2", options=[options[1]]),
        ]

        multi_option_extraction_data = MultiOptionExtractionData(
            multilingual=False, multi_value=True, options=options, samples=samples
        )

        multi_option_extractor.create_model(multi_option_extraction_data=multi_option_extraction_data)
        predictions = multi_option_extractor.get_multi_option_predictions(["option 1 option 2"])
        self.assertEqual([options[0], options[1]], predictions[0].options)

    def test_get_predictions_semantics(self):
        multi_option_extractor = MultiOptionExtractor(
            tenant=TestMultiOptionExtractor.TENANT, property_name=TestMultiOptionExtractor.PROPERTY_NAME
        )
        options = [Option(id="id1", label="Germany"), Option(id="id2", label="France")]
        samples = [
            MultiOptionExtractionSample(text="Berlin", options=[options[0]]),
            MultiOptionExtractionSample(text="Paris", options=[options[1]]),
        ] * 30

        multi_option_extraction_data = MultiOptionExtractionData(
            multilingual=True, multi_value=False, options=options, samples=samples
        )

        model_created, error = multi_option_extractor.create_model(multi_option_extraction_data=multi_option_extraction_data)
        predictions = multi_option_extractor.get_multi_option_predictions(["Paris", "Berlin"])
        self.assertTrue(model_created)
        self.assertEqual("", error)
        self.assertTrue(2, len(predictions))
        self.assertEqual([options[1]], predictions[0].options)
        self.assertEqual([options[0]], predictions[1].options)

    def test_get_predictions_semantics_multi_value(self):
        multi_option_extractor = MultiOptionExtractor(
            tenant=TestMultiOptionExtractor.TENANT, property_name=TestMultiOptionExtractor.PROPERTY_NAME
        )

        options = [Option(id="id1", label="Germany"), Option(id="id2", label="France")]
        samples = [
            MultiOptionExtractionSample(text="Berlin", options=[options[0]]),
            MultiOptionExtractionSample(text="Paris Berlin", options=[options[1], options[0]]),
        ] * 10

        multi_option_extraction_data = MultiOptionExtractionData(
            multilingual=False, multi_value=True, options=options, samples=samples
        )

        model_created, error = multi_option_extractor.create_model(multi_option_extraction_data=multi_option_extraction_data)
        predictions = multi_option_extractor.get_multi_option_predictions(["Paris Berlin", "Berlin"])
        self.assertTrue(model_created)
        self.assertEqual("", error)
        self.assertTrue(2, len(predictions))
        self.assertEqual([options[1], options[0]], predictions[0].options)
        self.assertEqual([options[0]], predictions[1].options)

    def test_get_predictions_semantics_similar_options_labels(self):
        multi_option_extractor = MultiOptionExtractor(
            tenant=TestMultiOptionExtractor.TENANT, property_name=TestMultiOptionExtractor.PROPERTY_NAME
        )

        options = [Option(id="id1", label="switzerland"), Option(id="id2", label="switzerland and spain")]
        samples = [
            MultiOptionExtractionSample(text="Two European countries", options=[options[1]]),
        ] * 6

        multi_option_extraction_data = MultiOptionExtractionData(
            multilingual=False, multi_value=False, options=options, samples=samples
        )

        model_created, error = multi_option_extractor.create_model(multi_option_extraction_data=multi_option_extraction_data)
        predictions = multi_option_extractor.get_multi_option_predictions(["Two European countries"])
        self.assertTrue(model_created)
        self.assertEqual("", error)
        self.assertTrue(1, len(predictions))
        self.assertEqual([options[1]], predictions[0].options)
