from unittest import TestCase

from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.methods.RegexSubtractionMethod import RegexSubtractionMethod


class TestRegexMethod(TestCase):
    def test_performance_text_in_front(self):
        semantic_information_data = [SemanticExtractionData(text="two", segment_text="one two", language_iso="en")]
        semantic_information_data += [SemanticExtractionData(text="three", segment_text="one three", language_iso="en")]
        regex_method = RegexSubtractionMethod("regex_subtraction_test", "regex_subtraction_test")

        self.assertEqual(100, regex_method.performance(semantic_information_data, 1)[0])

    def test_performance_text_in_back(self):
        semantic_information_data = [SemanticExtractionData(text="two", segment_text="two other", language_iso="en")]
        semantic_information_data += [SemanticExtractionData(text="three", segment_text="three other", language_iso="en")]
        regex_method = RegexSubtractionMethod("regex_subtraction_test", "regex_subtraction_test")

        self.assertEqual(100, regex_method.performance(semantic_information_data, 1)[0])

    def test_performance(self):
        semantic_information_data = [SemanticExtractionData(text="two", segment_text="two", language_iso="en")]
        semantic_information_data += [
            SemanticExtractionData(
                text="""Angola, Argentina, Austria, Benin, Botswana, Brazil, Burkina Faso, Chile, Congo, Costa Rica, Côte d’Ivoire, Czech Republic, Ecuador, Estonia, Ethiopia, Gabon, Germany, Guatemala, India, Indonesia, Ireland, Italy, Japan, Kazakhstan, Kenya, Libya, Malaysia, Maldives, Montenegro, Pakistan, Peru, Philippines, Poland, Republic of Korea, Republic of Moldova, Romania, Sierra Leone, Spain, Switzerland, Thailand, Uganda, Venezuela (Bolivarian Republic of)""",
                segment_text="""Angola, Argentina, Austria, Benin, Botswana, Brazil, Burkina Faso, Chile,
Congo, Costa Rica, Côte d’Ivoire, Czech Republic, Ecuador, Estonia,
Ethiopia, Gabon, Germany, Guatemala, India, Indonesia, Ireland, Italy,
Japan, Kazakhstan, Kenya, Libya, Malaysia, Maldives, Montenegro,
Pakistan, Peru, Philippines, Poland, Republic of Korea, Republic of
Moldova, Romania, Sierra Leone, Spain, Switzerland, Thailand, Uganda,
Venezuela (Bolivarian Republic of)""",
                language_iso="en",
            )
        ]
        regex_method = RegexSubtractionMethod("regex_subtraction_test", "regex_subtraction_test")

        self.assertEqual(100, regex_method.performance(semantic_information_data, 1)[0])

    def test_performance_text_front_and_back(self):
        semantic_information_data = [SemanticExtractionData(text="two", segment_text="1/ two final", language_iso="en")]
        semantic_information_data += [SemanticExtractionData(text="three", segment_text="1/ three final", language_iso="en")]
        regex_method = RegexSubtractionMethod("regex_subtraction_test", "regex_subtraction_test")

        self.assertEqual(100, regex_method.performance(semantic_information_data, 1)[0])

    def test_predict(self):
        semantic_information_data = [SemanticExtractionData(text="foo", segment_text="1/ foo end", language_iso="")]
        semantic_information_data += [SemanticExtractionData(text="var", segment_text="2/ var end", language_iso="")]
        regex_method = RegexSubtractionMethod("regex_subtraction_test", "regex_subtraction_test")

        regex_method.train(semantic_information_data)
        predictions = regex_method.predict(["3/ one end", "4/ two end"])
        self.assertEqual(2, len(predictions))
        self.assertEqual("one", predictions[0])
        self.assertEqual("two", predictions[1])

    def test_predict_void(self):
        semantic_information_data = [SemanticExtractionData(text="124", segment_text="", language_iso="")]
        semantic_information_data += [SemanticExtractionData(text="344", segment_text="", language_iso="")]
        regex_method = RegexSubtractionMethod("regex_subtraction_test", "regex_subtraction_test")

        regex_method.train(semantic_information_data)
        predictions = regex_method.predict([""])
        self.assertEqual(1, len(predictions))
        self.assertEqual("", predictions[0])
