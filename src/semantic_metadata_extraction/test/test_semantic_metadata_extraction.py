import os
import shutil
from os.path import join
from pathlib import Path
from unittest import TestCase

from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.SemanticMetadataExtraction import SemanticMetadataExtraction

DOCKER_VOLUME_PATH = join(Path(__file__).parent, "..", "..", "..", "docker_volume")


class TestSemanticMetadataExtraction(TestCase):
    def test_predictions_one_sample(self):
        tenant = "semantic_tenant_to_be_removed"
        property_name = "property_name"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        semantic_metadata_extraction = SemanticMetadataExtraction(tenant=tenant, property_name=property_name)

        semantic_information_data = [SemanticExtractionData(text="one", segment_text="two", language_iso="en")]
        semantic_metadata_extraction.create_model(semantic_information_data)
        predictions = semantic_metadata_extraction.get_semantic_predictions(["test 1", "test 2", "test 3"])

        self.assertEqual(["test 1", "test 2", "test 3"], predictions)

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

    def test_predictions_two_samples(self):
        tenant = "semantic_tenant_to_be_removed"
        property_name = "property_name"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        semantic_metadata_extraction = SemanticMetadataExtraction(tenant=tenant, property_name=property_name)

        semantic_information_data = [
            SemanticExtractionData(text="one", segment_text="two", language_iso="en"),
            SemanticExtractionData(text="one", segment_text="two", language_iso="en"),
        ]
        semantic_metadata_extraction.create_model(semantic_information_data)
        predictions = semantic_metadata_extraction.get_semantic_predictions(["one"])

        self.assertEqual(1, len(predictions))

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

    def test_predictions_same_input_output(self):
        tenant = "semantic_tenant_to_be_removed"
        property_name = "property_name"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        semantic_metadata_extraction = SemanticMetadataExtraction(tenant=tenant, property_name=property_name)

        semantic_information_data = [
            SemanticExtractionData(text="one", segment_text="one", language_iso="fr") for _ in range(2)
        ]
        semantic_metadata_extraction.create_model(semantic_information_data)
        predictions = semantic_metadata_extraction.get_semantic_predictions(["test 1", "test 2", "test 3"])

        self.assertEqual(["test 1", "test 2", "test 3"], predictions)

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

    def test_predictions_regex(self):
        tenant = "semantic_tenant_to_be_removed"
        property_name = "property_name"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        semantic_metadata_extraction = SemanticMetadataExtraction(tenant=tenant, property_name=property_name)

        semantic_information_data = [
            SemanticExtractionData(text="one", segment_text="one two", language_iso="en") for _ in range(5)
        ]
        semantic_metadata_extraction.create_model(semantic_information_data)
        predictions = semantic_metadata_extraction.get_semantic_predictions(["one two", "one three", "four"])

        self.assertTrue(os.path.exists(f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/RegexMethod"))
        self.assertEqual(["one", "one", ""], predictions)

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)
