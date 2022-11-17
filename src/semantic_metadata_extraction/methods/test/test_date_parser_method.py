from unittest import TestCase

from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.methods.DateParserMethod import DateParserMethod


class TestRegexMethod(TestCase):
    def test_performance(self):
        semantic_information_data = [
            SemanticExtractionData(text="5/13/1981", segment_text="13 May 1981", language_iso="en") for _ in range(6)
        ]
        date_parser_method = DateParserMethod("date_test", "date_test")
        self.assertEqual(100, date_parser_method.performance(semantic_information_data, 3)[0])

    def test_predict(self):
        semantic_information_data = [SemanticExtractionData(text="6/5/1982", segment_text="5 Jun 1982", language_iso="en")]
        date_parser_method = DateParserMethod("date_test", "date_test")
        date_parser_method.train(semantic_information_data)
        predictions = date_parser_method.predict(["5 Jun 1982"])
        self.assertEqual(["6/5/1982"], predictions)

    def test_performance_no_dates(self):
        semantic_information_data = [
            SemanticExtractionData(text="5/13/1981", segment_text="13 May 1981", language_iso="en"),
            SemanticExtractionData(text="text", segment_text="13 May 81", language_iso="en"),
        ]
        date_parser_method = DateParserMethod("date_test", "date_test")
        self.assertEqual(0, date_parser_method.performance(semantic_information_data, 3)[0])
