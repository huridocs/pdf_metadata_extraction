from unittest import TestCase

from data.SemanticExtractionData import SemanticExtractionData
from semantic_metadata_extraction.methods.DateParserMethod import DateParserMethod


class TestRegexMethod(TestCase):
    def test_performance(self):
        semantic_information_data = [
            SemanticExtractionData(text="13 May 1981", segment_text="13 May 1981", language_iso="en") for _ in range(6)
        ]
        date_parser_method = DateParserMethod("date_test", "date_test")
        self.assertEqual(100, date_parser_method.performance(semantic_information_data, 3)[0])

    # def test_performance_0(self):
    #     semantic_information_data = [
    #         SemanticExtractionData(text="12", segment_text="one two", language_iso="en") for _ in range(6)
    #     ]
    #     date_parser_method = RegexMethod("regex_test", "regex_test")
    #     self.assertEqual(0, date_parser_method.performance(semantic_information_data, 3)[0])
    #
    # def test_performance_50(self):
    #     semantic_information_data = [
    #         SemanticExtractionData(text="12", segment_text="one 12", language_iso="en") for _ in range(3)
    #     ]
    #
    #     semantic_information_data += [
    #         SemanticExtractionData(text="no regex", segment_text="one two", language_iso="en") for _ in range(1)
    #     ]
    #     date_parser_method = RegexMethod("regex_test", "regex_test")
    #
    #     self.assertEqual(50, date_parser_method.performance(semantic_information_data, 3)[0])
    #
    # def test_performance_no_samples(self):
    #     date_parser_method = RegexMethod("regex_test", "regex_test")
    #
    #     self.assertEqual(0, date_parser_method.performance([], 3))
    #
    # def test_performance_one_sample(self):
    #     semantic_information_data = [SemanticExtractionData(text="12", segment_text="one 12", language_iso="en")]
    #     date_parser_method = RegexMethod("regex_test", "regex_test")
    #
    #     self.assertEqual(0, date_parser_method.performance(semantic_information_data, 3)[0])
    #
    # def test_predict(self):
    #     semantic_information_data = [SemanticExtractionData(text="12", segment_text="", language_iso="")]
    #     semantic_information_data += [SemanticExtractionData(text="34", segment_text="", language_iso="")]
    #     date_parser_method = RegexMethod("regex_test", "regex_test")
    #
    #     date_parser_method.train(semantic_information_data)
    #     predictions = date_parser_method.predict(["one 12", "13", "14 foo"])
    #     self.assertEqual(3, len(predictions))
    #     self.assertEqual("12", predictions[0])
    #     self.assertEqual("13", predictions[1])
    #     self.assertEqual("14", predictions[2])
    #
    # def test_predict_void(self):
    #     semantic_information_data = [SemanticExtractionData(text="124", segment_text="", language_iso="")]
    #     semantic_information_data += [SemanticExtractionData(text="344", segment_text="", language_iso="")]
    #     date_parser_method = RegexMethod("regex_test", "regex_test")
    #
    #     date_parser_method.train(semantic_information_data)
    #     predictions = date_parser_method.predict([""])
    #     self.assertEqual(1, len(predictions))
    #     self.assertEqual("", predictions[0])
    #
    # def test_retrain(self):
    #     semantic_information_data = [SemanticExtractionData(text="1", segment_text="", language_iso="")]
    #     date_parser_method = RegexMethod("regex_test", "regex_test")
    #     date_parser_method.train(semantic_information_data)
    #
    #     semantic_information_data = [SemanticExtractionData(text="111", segment_text="", language_iso="")]
    #
    #     date_parser_method.train(semantic_information_data)
    #
    #     predictions = date_parser_method.predict(["111"])
    #     self.assertEqual(1, len(predictions))
    #     self.assertEqual("111", predictions[0])
