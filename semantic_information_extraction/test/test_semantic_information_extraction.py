import os
import shutil
from unittest import TestCase

from data.SemanticExtractionData import SemanticExtractionData
from semantic_information_extraction.SemanticInformationExtraction import SemanticInformationExtraction

DOCKER_VOLUME_PATH = '../../docker_volume'


class TestSemanticInformationExtraction(TestCase):
    def test_create_model(self):
        tenant = 'tenant_to_be_removed'
        extraction_name = 'extraction_name'

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

        semantic_information_extraction = SemanticInformationExtraction(tenant=tenant, extraction_name=extraction_name)

        semantic_information_data = [SemanticExtractionData(text="one", segment_text="one two")]
        semantic_information_extraction.create_model(semantic_information_data)

        self.assertTrue(os.path.exists(f'{DOCKER_VOLUME_PATH}/{tenant}/{extraction_name}/semantic_model/best_model'))

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

    def test_get_suggestions(self):
        tenant = 'tenant_to_be_removed'
        extraction_name = 'extraction_name'

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

        semantic_information_extraction = SemanticInformationExtraction(tenant=tenant, extraction_name=extraction_name)

        training_data = [SemanticExtractionData(text="English", segment_text="Origin: English") for _ in range(10)]
        semantic_extraction_input = training_data
        semantic_information_extraction.create_model(semantic_extraction_input)
        predicted_texts = semantic_information_extraction.get_semantic_predictions(["Origin: English"])

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

        self.assertEqual("English", predicted_texts[0])


