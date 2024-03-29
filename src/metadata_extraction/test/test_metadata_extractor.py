from unittest import TestCase

from data.ExtractionIdentifier import ExtractionIdentifier
from metadata_extraction.PdfMetadataExtractor import PdfMetadataExtractor


class TestMetadataExtractor(TestCase):
    TENANT = "multi_option_extraction_test"
    extraction_id = "extraction_id"

    def test_no_prediction_data(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)

        metadata_extractor = PdfMetadataExtractor(extraction_identifier, [])
        predictions = metadata_extractor.get_metadata_predictions()

        self.assertEqual(0, len(predictions))
