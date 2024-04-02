from unittest import TestCase
from data.ExtractionIdentifier import ExtractionIdentifier
from data.LabeledData import LabeledData
from data.Option import Option
from data.PdfData import PdfData
from data.ExtractionData import ExtractionData
from data.ExtractionSample import ExtractionSample
from extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import PdfToMultiOptionExtractor


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
            ExtractionSample(pdf_data_1, LabeledData(values=[options[0]])),
            ExtractionSample(pdf_data_2, LabeledData(values=[options[1]])),
            ExtractionSample(pdf_data_3, LabeledData(values=[options[2]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=False, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = PdfToMultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        predictions = multi_option_extraction.get_suggestions([pdf_data_1, pdf_data_3])

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
            ExtractionSample(pdf_data_1, LabeledData(values=[options[0], options[1]])),
            ExtractionSample(pdf_data_2, LabeledData(values=[options[1]])),
            ExtractionSample(pdf_data_3, LabeledData(values=[options[2], options[0]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = PdfToMultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        suggestions = multi_option_extraction.get_suggestions([pdf_data_1, pdf_data_3])

        self.assertEqual(2, len(suggestions))
        self.assertTrue(Option(id="1", label="1") in suggestions[0].values)
        self.assertTrue(Option(id="2", label="2") in suggestions[0].values)
        self.assertTrue(Option(id="3", label="3") not in suggestions[0].values)
        self.assertTrue(Option(id="3", label="3") in suggestions[1].values)
        self.assertTrue(Option(id="2", label="2") not in suggestions[1].values)
        self.assertTrue(Option(id="1", label="1") in suggestions[1].values)

    def test_tf_idf(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_1 = PdfData.from_texts(["point one point two"])
        pdf_data_2 = PdfData.from_texts(["point two"])
        pdf_data_3 = PdfData.from_texts(["point three point one"])

        samples = [
            ExtractionSample(pdf_data_1, LabeledData(values=[options[0], options[1]])),
            ExtractionSample(pdf_data_2, LabeledData(values=[options[1]])),
            ExtractionSample(pdf_data_3, LabeledData(values=[options[2], options[0]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = PdfToMultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        suggestions = multi_option_extraction.get_suggestions([pdf_data_1, pdf_data_3])

        self.assertEqual(2, len(suggestions))
        self.assertTrue(Option(id="1", label="1") in suggestions[0].values)
        self.assertTrue(Option(id="2", label="2") in suggestions[0].values)
        self.assertTrue(Option(id="3", label="3") not in suggestions[0].values)
        self.assertTrue(Option(id="3", label="3") in suggestions[1].values)
        self.assertTrue(Option(id="2", label="2") not in suggestions[1].values)
        self.assertTrue(Option(id="1", label="1") in suggestions[1].values)

    def test_no_prediction_data(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_1 = PdfData.from_texts(["point one point two"])
        pdf_data_2 = PdfData.from_texts(["point two"])
        pdf_data_3 = PdfData.from_texts(["point three point one"])

        samples = [
            ExtractionSample(pdf_data_1, LabeledData(values=[options[0]])),
            ExtractionSample(pdf_data_2, LabeledData(values=[options[1]])),
            ExtractionSample(pdf_data_3, LabeledData(values=[options[2]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        multi_option_extraction = PdfToMultiOptionExtractor(extraction_identifier)
        multi_option_extraction.create_model(multi_option_data)

        suggestions = multi_option_extraction.get_suggestions([])

        self.assertEqual(0, len(suggestions))
