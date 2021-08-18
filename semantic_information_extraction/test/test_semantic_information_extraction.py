import os
import shutil
from unittest import TestCase

from data.SemanticExtractionData import SemanticExtractionData
from semantic_information_extraction.SemanticInformationExtraction import SemanticInformationExtraction

DOCKER_VOLUME_PATH = f'{os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))}/docker_volume'


class TestSemanticInformationExtraction(TestCase):
    def test_create_model(self):
        tenant = 'tenant_to_be_removed'
        extraction_name = 'extraction_name'

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

        semantic_information_extraction = SemanticInformationExtraction(tenant=tenant, extraction_name=extraction_name)

        semantic_information_data = [SemanticExtractionData(text="one", segment_text="one two", language_iso="en")]
        semantic_information_extraction.create_model(semantic_information_data)

        self.assertTrue(os.path.exists(f'{DOCKER_VOLUME_PATH}/{tenant}/{extraction_name}/semantic_model/best_model'))

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

    def test_get_suggestions(self):
        tenant = 'tenant_to_be_removed'
        extraction_name = 'ex'

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

        semantic_information_extraction = SemanticInformationExtraction(tenant=tenant, extraction_name=extraction_name)

        training_data = [SemanticExtractionData(text="English",
                                                segment_text="Origin: English",
                                                language_iso="en") for _ in range(10)]

        semantic_extraction_input = training_data
        semantic_information_extraction.create_model(semantic_extraction_input)
        predicted_texts = semantic_information_extraction.get_semantic_predictions(["Origin: English"])

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

        self.assertEqual("English", predicted_texts[0])
        self.assertEqual(1, len(predicted_texts))

    def test_remove_model_if_it_is_not_good(self):
        tenant = 'tenant_to_be_removed'
        extraction_name = 'ex'

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

        semantic_information_extraction = SemanticInformationExtraction(tenant=tenant, extraction_name=extraction_name)

        semantic_information_extraction.create_model([SemanticExtractionData(text="Origin: English",
                                                                             segment_text="Origin: English",
                                                                             language_iso="en")])
        predicted_texts = semantic_information_extraction.get_semantic_predictions(["Origin: English"])

        self.assertEqual("Origin: English", predicted_texts[0])
        self.assertFalse(os.path.exists(semantic_information_extraction.model_path))
        self.assertFalse(os.path.exists(semantic_information_extraction.multilingual_model_path))
        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

    def test_remove_model_if_it_is_not_good_multilingual(self):
        tenant = 'tenant_to_be_removed'
        extraction_name = 'ex'

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

        semantic_information_extraction = SemanticInformationExtraction(tenant=tenant, extraction_name=extraction_name)

        semantic_information_extraction.create_model([SemanticExtractionData(text="Origin: English",
                                                                             segment_text="Origin: English",
                                                                             language_iso="spa")])
        predicted_texts = semantic_information_extraction.get_semantic_predictions(["Origin: English"])

        self.assertEqual("Origin: English", predicted_texts[0])
        self.assertFalse(os.path.exists(semantic_information_extraction.model_path))
        self.assertFalse(os.path.exists(semantic_information_extraction.multilingual_model_path))
        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

    def test_get_suggestions_spanish(self):
        tenant = 'tenant_to_be_removed'
        extraction_name = 'spa'

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

        semantic_information_extraction = SemanticInformationExtraction(tenant=tenant, extraction_name=extraction_name)

        semantic_extraction_input = [SemanticExtractionData(segment_text="Septuagésimo quinto período de sesiones",
                                                            text="período",
                                                            language_iso="spa") for _ in range(45)]
        semantic_information_extraction.create_model(semantic_extraction_input)
        predicted_texts = semantic_information_extraction.get_semantic_predictions(
            ["Septuagésimo quinto período de sesiones"])

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

        self.assertEqual(1, len(predicted_texts))
        self.assertEqual("período", predicted_texts[0])