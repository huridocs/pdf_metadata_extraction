import os
import shutil
from os.path import join
from unittest import TestCase

from config import DATA_PATH
from data.ExtractionIdentifier import ExtractionIdentifier
from data.PdfTagData import PdfTagData
from data.SemanticExtractionData import SemanticExtractionData
from data.SemanticPredictionData import SemanticPredictionData
from extractors.text_to_text_extractor.TextToTextExtractor import TextToTextExtractor


class TestTextToTextExtractor(TestCase):
    def test_predictions_one_sample(self):
        tenant = "semantic_tenant_to_be_removed"
        extraction_id = "extraction_id"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

        extraction_identifier = ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        semantic_metadata_extraction = TextToTextExtractor(extraction_identifier=extraction_identifier)

        pdf_tags = [PdfTagData.from_text("two")]
        semantic_information_data = [SemanticExtractionData(text="one", pdf_tags=pdf_tags, language_iso="en")]
        semantic_metadata_extraction.create_model(semantic_information_data)
        semantic_predictions_data = SemanticPredictionData.from_texts(["test 1", "test 2", "test 3"])
        predictions = semantic_metadata_extraction.get_semantic_predictions(semantic_predictions_data)

        self.assertEqual(["test 1", "test 2", "test 3"], predictions)

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

    def test_predictions_two_samples(self):
        tenant = "semantic_tenant_to_be_removed"
        extraction_id = "extraction_id"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

        extraction_identifier = ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        semantic_metadata_extraction = TextToTextExtractor(extraction_identifier=extraction_identifier)

        pdf_tags = [PdfTagData.from_text("one two")]
        semantic_information_data = [
            SemanticExtractionData(text="one", pdf_tags=pdf_tags, language_iso="en"),
            SemanticExtractionData(text="one", pdf_tags=pdf_tags, language_iso="en"),
        ]
        semantic_metadata_extraction.create_model(semantic_information_data)
        predictions = semantic_metadata_extraction.get_semantic_predictions(SemanticPredictionData.from_texts(["one"]))

        self.assertEqual(1, len(predictions))

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

    def test_predictions_same_input_output(self):
        tenant = "semantic_tenant_to_be_removed"
        extraction_id = "extraction_id"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

        extraction_identifier = ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        semantic_metadata_extraction = TextToTextExtractor(extraction_identifier=extraction_identifier)

        pdf_tags = [PdfTagData.from_text("one")]

        semantic_information_data = [
            SemanticExtractionData(text="one", pdf_tags=pdf_tags, language_iso="fr") for _ in range(2)
        ]
        semantic_metadata_extraction.create_model(semantic_information_data)
        semantic_predictions_data = SemanticPredictionData.from_texts(["test 1", "test 2", "test 3"])
        predictions = semantic_metadata_extraction.get_semantic_predictions(semantic_predictions_data)

        self.assertEqual(["test 1", "test 2", "test 3"], predictions)

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

    def test_predictions_regex(self):
        tenant = "semantic_tenant_to_be_removed"
        extraction_id = "extraction_id"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

        extraction_identifier = ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        semantic_metadata_extraction = TextToTextExtractor(extraction_identifier=extraction_identifier)

        pdf_tags = [PdfTagData.from_text("one two")]
        semantic_information_data = [
            SemanticExtractionData(text="one", pdf_tags=pdf_tags, language_iso="en") for _ in range(5)
        ]
        semantic_metadata_extraction.create_model(semantic_information_data)
        semantic_predictions_data = SemanticPredictionData.from_texts(["one two", "one three", "four"])

        predictions = semantic_metadata_extraction.get_semantic_predictions(semantic_predictions_data)

        self.assertTrue(os.path.exists(f"{DATA_PATH}/{tenant}/{extraction_id}/RegexMethod"))
        self.assertEqual(["one", "one", ""], predictions)

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
