from unittest import TestCase

from data.PdfTagData import PdfTagData
from data.SemanticExtractionData import SemanticExtractionData
from data.SemanticPredictionData import SemanticPredictionData
from semantic_metadata_extraction.methods.DateParserWithBreaksMethod import DateParserWithBreaksMethod


class TestDateParserWithBreaksMethod(TestCase):
    def test_predict(self):
        date_parser_method = DateParserWithBreaksMethod("date_breaks_test", "date_breaks_test")

        pdf_tags_data = [PdfTagData.from_text("5 Jun 1982")]
        semantic_information_data = [SemanticExtractionData(text="1982-06-05", pdf_tags=pdf_tags_data, language_iso="en")]
        date_parser_method.train(semantic_information_data)

        pdf_tags_data_1 = [
            PdfTagData.from_text("ORDER OF THE INTER-AMERICAN COURT OF HUMAN RIGHTS 1"),
            PdfTagData.from_text("OF FEBRUARY 9, 2006"),
        ]

        pdf_tags_data_2 = [
            PdfTagData.from_text("ORDER OF THE INTER-AMERICAN COURT OF HUMAN RIGHTS 1"),
            PdfTagData.from_text("OF MARCH 10, 2007"),
        ]
        predictions = date_parser_method.predict(
            [SemanticPredictionData(pdf_tags_data=pdf_tags_data_1), SemanticPredictionData(pdf_tags_data=pdf_tags_data_2)]
        )
        self.assertEqual("2006-02-09", predictions[0])
        self.assertEqual("2007-03-10", predictions[1])
