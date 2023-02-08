from unittest import TestCase

from data.PdfTagData import PdfTagData
from data.SemanticExtractionData import SemanticExtractionData
from data.SemanticPredictionData import SemanticPredictionData
from semantic_metadata_extraction.methods.SameInputOutputMethod import SameInputOutputMethod


class TestSameInputMethod(TestCase):
    def test_performance_100(self):
        pdf_tag_data = PdfTagData.from_text("a b c")
        semantic_information_data = [SemanticExtractionData(text="a b c", pdf_tags=[pdf_tag_data], language_iso="en")]
        same_input_output_method = SameInputOutputMethod("same_input_test", "same_input_test")
        self.assertEqual(100, same_input_output_method.performance(semantic_information_data, 2)[0])

    def test_performance_50(self):
        semantic_extraction_1 = SemanticExtractionData(
            text="a b c", pdf_tags=[PdfTagData.from_text("a b c")], language_iso="en"
        )
        semantic_extraction_2 = SemanticExtractionData(text="2", pdf_tags=[PdfTagData.from_text("a b c")], language_iso="en")
        semantic_information_data = [semantic_extraction_1, semantic_extraction_2]
        same_input_output_method = SameInputOutputMethod("same_input_test", "same_input_test")
        self.assertEqual(50, same_input_output_method.performance(semantic_information_data, 2)[0])

    def test_predict(self):
        same_input_output_method = SameInputOutputMethod("same_input_test", "same_input_test")
        predictions = same_input_output_method.predict([SemanticPredictionData.from_text("test 1")])
        self.assertEqual(["test 1"], predictions)
