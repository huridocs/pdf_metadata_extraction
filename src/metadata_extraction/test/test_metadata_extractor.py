from unittest import TestCase
from data.ExtractionIdentifier import ExtractionIdentifier
from data.Option import Option
from metadata_extraction.PdfData import PdfData
from multi_option_extraction.data.MultiOptionData import MultiOptionData
from multi_option_extraction.data.MultiOptionSample import MultiOptionSample
from multi_option_extraction.MultiOptionExtractor import MultiOptionExtractor


class TestMultiOptionExtraction(TestCase):
    TENANT = "multi_option_extraction_test"
    extraction_id = "extraction_id"

    def test_single_value(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_1 = PdfData.from_texts(["point 1"])
        pdf_data_2 = PdfData.from_texts(["point 2"])
        pdf_data_3 = PdfData.from_texts(["point 3"])

        samples = [
            MultiOptionSample(pdf_data_1, [options[0]], "en"),
            MultiOptionSample(pdf_data_2, [options[1]], "en"),
            MultiOptionSample(pdf_data_3, [options[2]], "en"),
        ]

        multi_option_data = MultiOptionData(
            multi_value=False, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = MultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        predictions = multi_option_extraction.get_multi_option_predictions([pdf_data_1, pdf_data_3])

        self.assertEqual(2, len(predictions))
        self.assertEqual([Option(id="1", label="1")], predictions[0].values)
        self.assertEqual([Option(id="3", label="3")], predictions[1].values)

    def test_multi_value(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_1 = PdfData.from_texts(["point 1 point 2"])
        pdf_data_2 = PdfData.from_texts(["point 2"])
        pdf_data_3 = PdfData.from_texts(["point 3 point 1"])

        samples = [
            MultiOptionSample(pdf_data_1, [options[0], options[1]], "en"),
            MultiOptionSample(pdf_data_2, [options[1]], "en"),
            MultiOptionSample(pdf_data_3, [options[2], options[0]], "en"),
        ]

        multi_option_data = MultiOptionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = MultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        predictions = multi_option_extraction.get_multi_option_predictions([pdf_data_1, pdf_data_3])

        self.assertEqual(2, len(predictions))
        self.assertTrue(Option(id="1", label="1") in predictions[0].values)
        self.assertTrue(Option(id="2", label="2") in predictions[0].values)
        self.assertTrue(Option(id="3", label="3") not in predictions[0].values)
        self.assertTrue(Option(id="3", label="3") in predictions[1].values)
        self.assertTrue(Option(id="2", label="2") not in predictions[1].values)
        self.assertTrue(Option(id="1", label="1") in predictions[1].values)

    def test_tf_idf(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_1 = PdfData.from_texts(["point one point two"])
        pdf_data_2 = PdfData.from_texts(["point two"])
        pdf_data_3 = PdfData.from_texts(["point three point one"])

        samples = [
            MultiOptionSample(pdf_data_1, [options[0], options[1]], "en"),
            MultiOptionSample(pdf_data_2, [options[1]], "en"),
            MultiOptionSample(pdf_data_3, [options[2], options[0]], "en"),
        ]

        multi_option_data = MultiOptionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = MultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        predictions = multi_option_extraction.get_multi_option_predictions([pdf_data_1, pdf_data_3])

        self.assertEqual(2, len(predictions))
        self.assertTrue(Option(id="1", label="1") in predictions[0].values)
        self.assertTrue(Option(id="2", label="2") in predictions[0].values)
        self.assertTrue(Option(id="3", label="3") not in predictions[0].values)
        self.assertTrue(Option(id="3", label="3") in predictions[1].values)
        self.assertTrue(Option(id="2", label="2") not in predictions[1].values)
        self.assertTrue(Option(id="1", label="1") in predictions[1].values)

    def test_no_prediction_data(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_1 = PdfData.from_texts(["point one point two"])
        pdf_data_2 = PdfData.from_texts(["point two"])
        pdf_data_3 = PdfData.from_texts(["point three point one"])

        samples = [
            MultiOptionSample(pdf_data_1, [options[0], options[1]], "en"),
            MultiOptionSample(pdf_data_2, [options[1]], "en"),
            MultiOptionSample(pdf_data_3, [options[2], options[0]], "en"),
        ]

        multi_option_data = MultiOptionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = MultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        predictions = multi_option_extraction.get_multi_option_predictions([])

        self.assertEqual(0, len(predictions))
