from unittest import TestCase

from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.methods.RegexMethod import RegexMethod


class TestRegexMethod(TestCase):
    def test_performance_100(self):
        semantic_information_data = [
            SemanticExtractionData(text="12", segment_text="one 12", language_iso="en") for _ in range(6)
        ]
        regex_method = RegexMethod("regex_test", "regex_test")
        self.assertEqual(100, regex_method.performance(semantic_information_data, 3)[0])

    def test_performance_0(self):
        semantic_information_data = [
            SemanticExtractionData(text="12", segment_text="one two", language_iso="en") for _ in range(6)
        ]
        regex_method = RegexMethod("regex_test", "regex_test")
        self.assertEqual(0, regex_method.performance(semantic_information_data, 3)[0])

    def test_performance_50(self):
        semantic_information_data = [
            SemanticExtractionData(text="12", segment_text="one 12", language_iso="en") for _ in range(3)
        ]

        semantic_information_data += [
            SemanticExtractionData(text="no regex", segment_text="one two", language_iso="en") for _ in range(1)
        ]
        regex_method = RegexMethod("regex_test", "regex_test")

        self.assertEqual(50, regex_method.performance(semantic_information_data, 3)[0])

    def test_performance_no_samples(self):
        regex_method = RegexMethod("regex_test", "regex_test")

        self.assertEqual(0, regex_method.performance([], 3))

    def test_performance_one_sample(self):
        semantic_information_data = [SemanticExtractionData(text="12", segment_text="one 12", language_iso="en")]
        regex_method = RegexMethod("regex_test", "regex_test")

        self.assertEqual(0, regex_method.performance(semantic_information_data, 3)[0])

    def test_predict(self):
        semantic_information_data = [SemanticExtractionData(text="12", segment_text="", language_iso="")]
        semantic_information_data += [SemanticExtractionData(text="34", segment_text="", language_iso="")]
        regex_method = RegexMethod("regex_test", "regex_test")

        regex_method.train(semantic_information_data)
        predictions = regex_method.predict(["one 12", "13", "14 foo"])
        self.assertEqual(3, len(predictions))
        self.assertEqual("12", predictions[0])
        self.assertEqual("13", predictions[1])
        self.assertEqual("14", predictions[2])

    def test_predict_void(self):
        semantic_information_data = [SemanticExtractionData(text="124", segment_text="", language_iso="")]
        semantic_information_data += [SemanticExtractionData(text="344", segment_text="", language_iso="")]
        regex_method = RegexMethod("regex_test", "regex_test")

        regex_method.train(semantic_information_data)
        predictions = regex_method.predict([""])
        self.assertEqual(1, len(predictions))
        self.assertEqual("", predictions[0])

    def test_retrain(self):
        semantic_information_data = [SemanticExtractionData(text="1", segment_text="", language_iso="")]
        regex_method = RegexMethod("regex_test", "regex_test")
        regex_method.train(semantic_information_data)

        semantic_information_data = [SemanticExtractionData(text="111", segment_text="", language_iso="")]

        regex_method.train(semantic_information_data)

        predictions = regex_method.predict(["111"])
        self.assertEqual(1, len(predictions))
        self.assertEqual("111", predictions[0])
