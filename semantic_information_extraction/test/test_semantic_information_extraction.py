import os
import shutil
import time
from unittest import TestCase

from data.SemanticExtractionData import SemanticExtractionData
from semantic_information_extraction.SemanticInformationExtraction import SemanticInformationExtraction

DOCKER_VOLUME_PATH = f'{os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))}/docker_volume'


class TestSemanticInformationExtraction(TestCase):
    def test_create_model(self):
        tenant = 'semantic_tenant_to_be_removed_1'
        extraction_name = 'extraction_name'

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

        semantic_information_extraction = SemanticInformationExtraction(tenant=tenant, extraction_name=extraction_name)

        semantic_information_data = [SemanticExtractionData(text="one", segment_text="one two", language_iso="en")]
        semantic_information_extraction.create_model(semantic_information_data)
        print(f'{DOCKER_VOLUME_PATH}/{tenant}/{extraction_name}/semantic_model/best_model')
        print(f'{DOCKER_VOLUME_PATH}/{tenant}/{extraction_name}/semantic_model/best_model')
        self.assertTrue(os.path.exists(f'{DOCKER_VOLUME_PATH}/{tenant}/{extraction_name}/semantic_model/best_model'))
        time.sleep(5)
        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

    def test_avoid_wrong_predictions(self):
        tenant = 'semantic_tenant_to_be_removed'
        extraction_name = 'ex'

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

        semantic_information_extraction = SemanticInformationExtraction(tenant=tenant, extraction_name=extraction_name)

        semantic_information_extraction.create_model([SemanticExtractionData(text="English",
                                                                             segment_text="Origin: English",
                                                                             language_iso="spa")])
        predicted_texts = semantic_information_extraction.get_semantic_predictions(["Origin: English"])

        self.assertEqual("Origin: English", predicted_texts[0])
        self.assertFalse(os.path.exists(semantic_information_extraction.model_path))
        self.assertFalse(os.path.exists(semantic_information_extraction.multilingual_model_path))

        shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)

    # def test_remove_model_if_it_is_not_good(self):
    #     tenant = 'semantic_tenant_to_be_removed_2'
    #     extraction_name = 'ex'
    #
    #     shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)
    #
    #     semantic_information_extraction = SemanticInformationExtraction(tenant=tenant, extraction_name=extraction_name)
    #
    #     semantic_information_extraction.create_model([SemanticExtractionData(text="Origin: English",
    #                                                                          segment_text="Origin: English",
    #                                                                          language_iso="en")])
    #     predicted_texts = semantic_information_extraction.get_semantic_predictions(["Origin: English"])
    #
    #     self.assertEqual("Origin: English", predicted_texts[0])
    #     self.assertFalse(os.path.exists(semantic_information_extraction.model_path))
    #     self.assertFalse(os.path.exists(semantic_information_extraction.multilingual_model_path))
    #
    #     shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)
    #
    # def test_remove_model_if_it_is_not_good_multilingual(self):
    #     tenant = 'semantic_tenant_to_be_removed_3'
    #     extraction_name = 'ex'
    #
    #     shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)
    #
    #     semantic_information_extraction = SemanticInformationExtraction(tenant=tenant, extraction_name=extraction_name)
    #
    #     semantic_information_extraction.create_model([SemanticExtractionData(text="Origin: English",
    #                                                                          segment_text="Origin: English",
    #                                                                          language_iso="spa")])
    #     predicted_texts = semantic_information_extraction.get_semantic_predictions(["Origin: English"])
    #
    #     self.assertEqual("Origin: English", predicted_texts[0])
    #     self.assertFalse(os.path.exists(semantic_information_extraction.model_path))
    #     self.assertFalse(os.path.exists(semantic_information_extraction.multilingual_model_path))
    #     shutil.rmtree(f'{DOCKER_VOLUME_PATH}/{tenant}', ignore_errors=True)