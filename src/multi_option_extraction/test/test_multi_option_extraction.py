from unittest import TestCase

from data.Option import Option
from multi_option_extraction.MultiOptionExtractor import MultiOptionExtractor


class TestMultiOptionExtraction(TestCase):
    TENANT = "multi_option_extraction_test"
    extraction_id = "extraction_id"

    def test_fuzzy(self):
        multi_option_extraction = MultiOptionExtractor(self.TENANT, self.extraction_id)
        multi_option_extraction.options = [Option(id="1", label="1"), Option(id="2", label="2")]
        options = multi_option_extraction.get_options_fuzzy("a12")

        self.assertEqual([Option(id="1", label="1")], options)

    def test_other_fuzzy(self):
        multi_option_extraction = MultiOptionExtractor(self.TENANT, self.extraction_id)
        multi_option_extraction.options = [Option(id="a", label="option a"), Option(id="b", label="option b")]
        options = multi_option_extraction.get_options_fuzzy("b")

        self.assertEqual([Option(id="b", label="option b")], options)

    def test_fuzzy_multi_value(self):
        multi_option_extraction = MultiOptionExtractor(self.TENANT, self.extraction_id)
        multi_option_extraction.options = [Option(id="1", label="1"), Option(id="2", label="2")]
        multi_option_extraction.multi_value = True
        options = multi_option_extraction.get_options_fuzzy("a12")

        self.assertEqual([Option(id="1", label="1"), Option(id="2", label="2")], options)
