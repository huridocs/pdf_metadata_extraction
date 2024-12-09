from unittest import TestCase

import mongomock
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier

from AsynchronousExtractor import AsynchronousExtractor
from data.ExtractionStatus import ExtractionStatus


class TestExtractionStatus(TestCase):

    def test_no_training_exists(self):
        extraction_identifier = ExtractionIdentifier(extraction_name="extraction_status_extractor")
        extraction_status: ExtractionStatus = AsynchronousExtractor(extraction_identifier).get_status()
        self.assertEqual(ExtractionStatus.NO_MODEL, extraction_status)

    def test_training_exists(self):
        extraction_identifier = ExtractionIdentifier(extraction_name="extraction_status_extractor")
        extraction_status: ExtractionStatus = AsynchronousExtractor(extraction_identifier).get_status()
        self.assertEqual(ExtractionStatus.NO_MODEL, extraction_status)
