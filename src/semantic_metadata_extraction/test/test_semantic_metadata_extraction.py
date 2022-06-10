import os
import shutil
from os.path import join
from pathlib import Path
from unittest import TestCase

from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.SemanticMetadataExtraction import SemanticMetadataExtraction

DOCKER_VOLUME_PATH = join(Path(__file__).parent, "..", "..", "..", "docker_volume")


class TestSemanticMetadataExtraction(TestCase):
    def test_create_model(self):
        tenant = "semantic_tenant_to_be_removed"
        property_name = "property_name"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        semantic_metadata_extraction = SemanticMetadataExtraction(tenant=tenant, property_name=property_name)

        semantic_information_data = [
            SemanticExtractionData(text="one", segment_text="one two", language_iso="en") for _ in range(5)
        ]
        semantic_metadata_extraction.create_model(semantic_information_data)

        self.assertTrue(os.path.exists(f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/semantic_model/best_model"))

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

    def test_remove_model_if_it_is_not_good(self):
        tenant = "semantic_tenant_to_be_removed"
        property_name = "ex"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        semantic_metadata_extraction = SemanticMetadataExtraction(tenant=tenant, property_name=property_name)

        semantic_metadata_extraction.create_model(
            [
                SemanticExtractionData(
                    text="Origin: English",
                    segment_text="Origin: English",
                    language_iso="en",
                )
            ]
        )
        predicted_texts = semantic_metadata_extraction.get_semantic_predictions(["Origin: English"])

        self.assertEqual("Origin: English", predicted_texts[0])
        self.assertFalse(os.path.exists(semantic_metadata_extraction.model_path))
        self.assertFalse(os.path.exists(semantic_metadata_extraction.multilingual_model_path))

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

    def test_remove_model_if_it_is_not_good_multilingual(self):
        tenant = "semantic_tenant_to_be_removed"
        property_name = "ex"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        semantic_metadata_extraction = SemanticMetadataExtraction(tenant=tenant, property_name=property_name)

        semantic_metadata_extraction.create_model(
            [
                SemanticExtractionData(
                    text="Origin: English",
                    segment_text="Origin: English",
                    language_iso="spa",
                )
            ]
        )
        predicted_texts = semantic_metadata_extraction.get_semantic_predictions(["Origin: English"])

        self.assertEqual("Origin: English", predicted_texts[0])
        self.assertFalse(os.path.exists(semantic_metadata_extraction.model_path))
        self.assertFalse(os.path.exists(semantic_metadata_extraction.multilingual_model_path))
        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)
