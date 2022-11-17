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

    def test_predict_multiple_dates_spanish(self):
        text = "Informe no. 52/16 caso 12.521 fondo Maria Laura órdenes guerra y otros Chile 30 de noviembre de 2016"
        semantic_information_data = [SemanticExtractionData(text="30/11/2016", segment_text=text, language_iso="es")]
        date_parser_method = DateParserMethod("date_test", "date_test")
        date_parser_method.train(semantic_information_data)
        predictions = date_parser_method.predict([text])
        self.assertEqual(["11/30/2016"], predictions)

    def test_performance_no_dates(self):
        semantic_information_data = [
            SemanticExtractionData(text="5/13/1981", segment_text="13 May 1981", language_iso="en"),
            SemanticExtractionData(text="text", segment_text="13 May 81", language_iso="en"),
        ]
        date_parser_method = DateParserMethod("date_test", "date_test")
        self.assertEqual(0, date_parser_method.performance(semantic_information_data, 3)[0])
