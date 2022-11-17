from unittest import TestCase

from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.methods.DateParserMethod import DateParserMethod


class TestRegexMethod(TestCase):
    def test_performance(self):
        semantic_information_data = [
            SemanticExtractionData(text="13 May, 1981", segment_text="13 May 1981", language_iso="en") for _ in range(6)
        ]
        date_parser_method = DateParserMethod("date_test", "date_test")
        self.assertEqual(100, date_parser_method.performance(semantic_information_data, 3)[0])
