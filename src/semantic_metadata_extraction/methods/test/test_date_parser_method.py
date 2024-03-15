from unittest import TestCase

from data.ExtractionIdentifier import ExtractionIdentifier
from data.PdfTagData import PdfTagData
from data.SemanticExtractionData import SemanticExtractionData
from data.SemanticPredictionData import SemanticPredictionData
from semantic_metadata_extraction.methods.DateParserMethod import DateParserMethod


class TestDateParserMethod(TestCase):
    def test_performance(self):
        pdf_tags_data = [PdfTagData.from_text("13 May 1981")]
        semantic_information_data = [
            SemanticExtractionData(text="1981-05-13", pdf_tags=pdf_tags_data, language_iso="en") for _ in range(6)
        ]
        extraction_identifier = ExtractionIdentifier(run_name="date_test", extraction_name="date_test")
        date_parser_method = DateParserMethod(extraction_identifier)
        self.assertEqual(100, date_parser_method.performance(semantic_information_data, 3)[0])

    def test_predict(self):
        pdf_tags_data = [PdfTagData.from_text("5 Jun 1982")]
        semantic_information_data = [SemanticExtractionData(text="1982-06-05", pdf_tags=pdf_tags_data, language_iso="en")]
        extraction_identifier = ExtractionIdentifier(run_name="date_test", extraction_name="date_test")
        date_parser_method = DateParserMethod(extraction_identifier)
        date_parser_method.train(semantic_information_data)

        predictions = date_parser_method.predict([SemanticPredictionData.from_text("5 Jun 1982")])
        self.assertEqual(["1982-06-05"], predictions)

    def test_predict_multiple_dates_spanish(self):
        text = "Informe no. 52/16 caso 12.521 fondo Maria Laura órdenes guerra y otros Chile 30 de noviembre de 2016"
        pdf_tags_data = [PdfTagData.from_text(text)]
        semantic_information_data = [SemanticExtractionData(text="2016-11-30", pdf_tags=pdf_tags_data, language_iso="es")]
        extraction_identifier = ExtractionIdentifier(run_name="date_test", extraction_name="date_test")
        date_parser_method = DateParserMethod(extraction_identifier)
        date_parser_method.train(semantic_information_data)

        predictions = date_parser_method.predict([SemanticPredictionData.from_text(text)])
        self.assertEqual(["2016-11-30"], predictions)

    def test_performance_no_dates(self):
        semantic_information_data = [
            SemanticExtractionData(text="1981-05-13", pdf_tags=[PdfTagData.from_text("13 May 1981")], language_iso="en"),
            SemanticExtractionData(text="text", pdf_tags=[PdfTagData.from_text("13 May 81")], language_iso="en"),
        ]
        extraction_identifier = ExtractionIdentifier(run_name="date_test", extraction_name="date_test")
        date_parser_method = DateParserMethod(extraction_identifier)
        self.assertEqual(50, date_parser_method.performance(semantic_information_data, 3)[0])

    def test_performance_multiple_tags(self):
        pdf_tags_data = PdfTagData.from_texts(["13 May", "1981"])
        semantic_information_data = [
            SemanticExtractionData(text="1981-05-13", pdf_tags=pdf_tags_data, language_iso="en") for _ in range(6)
        ]
        extraction_identifier = ExtractionIdentifier(run_name="date_test", extraction_name="date_test")
        date_parser_method = DateParserMethod(extraction_identifier)
        self.assertEqual(100, date_parser_method.performance(semantic_information_data, 3)[0])
