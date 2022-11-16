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

    def test_predictions_t5(self):
        tenant = "semantic_tenant_to_be_removed"
        property_name = "other_property_name"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        semantic_metadata_extraction = SemanticMetadataExtraction(tenant=tenant, property_name=property_name)

        semantic_information_data = [
            SemanticExtractionData(text="Paris", segment_text="France", language_iso="en") for _ in range(3)
        ]
        semantic_metadata_extraction.create_model(semantic_information_data)
        predictions = semantic_metadata_extraction.get_semantic_predictions(["France"])

        self.assertTrue(os.path.exists(f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/MT5TrueCaseEnglishSpanishMethod"))
        self.assertEqual(["Paris"], predictions)

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

    def test_remove_previous_model(self):
        tenant = "semantic_tenant_to_be_removed"
        property_name = "remove_property_name"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        semantic_metadata_extraction = SemanticMetadataExtraction(tenant=tenant, property_name=property_name)

        semantic_information_data = [
            SemanticExtractionData(text="one", segment_text="one two", language_iso="en") for _ in range(3)
        ]
        semantic_metadata_extraction.create_model(semantic_information_data)

        semantic_information_data = [
            SemanticExtractionData(text="three", segment_text="one two", language_iso="en") for _ in range(3)
        ]

        semantic_metadata_extraction.create_model(semantic_information_data)

        predictions = semantic_metadata_extraction.get_semantic_predictions(["one two"])

        self.assertTrue(os.path.exists(f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/MT5TrueCaseEnglishSpanishMethod"))
        self.assertEqual(["three"], predictions)

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)
