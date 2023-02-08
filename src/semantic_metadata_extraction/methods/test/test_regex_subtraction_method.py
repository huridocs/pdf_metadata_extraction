from unittest import TestCase

from data.PdfTagData import PdfTagData
from data.SemanticExtractionData import SemanticExtractionData
from data.SemanticPredictionData import SemanticPredictionData
from semantic_metadata_extraction.methods.RegexSubtractionMethod import RegexSubtractionMethod


class TestRegexSubtractionMethod(TestCase):
    def test_performance_text_in_front(self):
        semantic_information_data = [
            SemanticExtractionData(text="two", pdf_tags=[PdfTagData.from_text("one two")], language_iso="en")
        ]
        semantic_information_data += [
            SemanticExtractionData(text="three", pdf_tags=[PdfTagData.from_text("one three")], language_iso="en")
        ]
        regex_method = RegexSubtractionMethod("regex_subtraction_test", "regex_subtraction_test")

        self.assertEqual(100, regex_method.performance(semantic_information_data, 1)[0])

    def test_performance_text_in_back(self):
        semantic_information_data = [
            SemanticExtractionData(text="two", pdf_tags=[PdfTagData.from_text("two other")], language_iso="en")
        ]
        semantic_information_data += [
            SemanticExtractionData(text="three", pdf_tags=[PdfTagData.from_text("three other")], language_iso="en")
        ]
        regex_method = RegexSubtractionMethod("regex_subtraction_test", "regex_subtraction_test")

        self.assertEqual(100, regex_method.performance(semantic_information_data, 1)[0])

    def test_performance(self):
        semantic_information_data = [
            SemanticExtractionData(text="two", pdf_tags=[PdfTagData.from_text("two")], language_iso="en")
        ]
        semantic_information_data += [
            SemanticExtractionData(
                text="""Angola, Argentina, Austria, Benin, Botswana, Brazil, Burkina Faso, Chile, Congo, Costa Rica, Côte d’Ivoire, Czech Republic, Ecuador, Estonia, Ethiopia, Gabon, Germany, Guatemala, India, Indonesia, Ireland, Italy, Japan, Kazakhstan, Kenya, Libya, Malaysia, Maldives, Montenegro, Pakistan, Peru, Philippines, Poland, Republic of Korea, Republic of Moldova, Romania, Sierra Leone, Spain, Switzerland, Thailand, Uganda, Venezuela (Bolivarian Republic of)""",
                pdf_tags=[
                    PdfTagData.from_text("Angola, Argentina, Austria, Benin, Botswana, Brazil, Burkina Faso, Chile,"),
                    PdfTagData.from_text("Congo, Costa Rica, Côte d’Ivoire, Czech Republic, Ecuador, Estonia,"),
                    PdfTagData.from_text("Ethiopia, Gabon, Germany, Guatemala, India, Indonesia, Ireland, Italy,"),
                    PdfTagData.from_text("Japan, Kazakhstan, Kenya, Libya, Malaysia, Maldives, Montenegro,"),
                    PdfTagData.from_text("Pakistan, Peru, Philippines, Poland, Republic of Korea, Republic of"),
                    PdfTagData.from_text("Moldova, Romania, Sierra Leone, Spain, Switzerland, Thailand, Uganda,"),
                    PdfTagData.from_text("Venezuela (Bolivarian Republic of)"),
                ],
                language_iso="en",
            )
        ]
        regex_method = RegexSubtractionMethod("regex_subtraction_test", "regex_subtraction_test")

        performance, _ = regex_method.performance(semantic_information_data, 1)
        self.assertEqual(100, performance)

    def test_performance_text_front_and_back(self):
        semantic_information_data = [
            SemanticExtractionData(text="two", pdf_tags=[PdfTagData.from_text("1/ two final")], language_iso="en")
        ]
        semantic_information_data += [
            SemanticExtractionData(text="three", pdf_tags=[PdfTagData.from_text("1/ three final")], language_iso="en")
        ]
        regex_method = RegexSubtractionMethod("regex_subtraction_test", "regex_subtraction_test")

        self.assertEqual(100, regex_method.performance(semantic_information_data, 1)[0])

    def test_predict(self):
        semantic_information_data = [
            SemanticExtractionData(text="foo", pdf_tags=[PdfTagData.from_text("1/ foo end")], language_iso="")
        ]
        semantic_information_data += [
            SemanticExtractionData(text="var", pdf_tags=[PdfTagData.from_text("2/ var end")], language_iso="")
        ]
        regex_method = RegexSubtractionMethod("regex_subtraction_test", "regex_subtraction_test")

        regex_method.train(semantic_information_data)
        predictions = regex_method.predict(
            [SemanticPredictionData.from_text("3/ one end"), SemanticPredictionData.from_text("4/ two end")]
        )
        self.assertEqual(2, len(predictions))
        self.assertEqual("one", predictions[0])
        self.assertEqual("two", predictions[1])

    def test_predict_void(self):
        semantic_information_data = [
            SemanticExtractionData(text="124", pdf_tags=[PdfTagData.from_text("")], language_iso="")
        ]
        semantic_information_data += [
            SemanticExtractionData(text="344", pdf_tags=[PdfTagData.from_text("")], language_iso="")
        ]
        regex_method = RegexSubtractionMethod("regex_subtraction_test", "regex_subtraction_test")

        regex_method.train(semantic_information_data)
        predictions = regex_method.predict([SemanticPredictionData.from_text("")])
        self.assertEqual(1, len(predictions))
        self.assertEqual("", predictions[0])
