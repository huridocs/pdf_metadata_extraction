from unittest import TestCase
from data.ExtractionIdentifier import ExtractionIdentifier
from data.LabeledData import LabeledData
from data.Option import Option
from data.PdfData import PdfData
from data.ExtractionData import ExtractionData
from data.PredictionSample import PredictionSample
from data.TrainingSample import TrainingSample
from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelectorFuzzyCommas import (
    FastSegmentSelectorFuzzyCommas,
)


class TestFastSegmentSelectorFuzzyCommas(TestCase):
    TENANT = "multi_option_extraction_test"
    extraction_id = "extraction_id"

    def test_performance_100(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_1 = PdfData.from_texts(["1, 2"])
        pdf_data_2 = PdfData.from_texts(["2"])
        pdf_data_3 = PdfData.from_texts(["3, 1"])
        pdf_data_4 = PdfData.from_texts(["2, 3"])

        samples = [
            TrainingSample(pdf_data_1, LabeledData(values=[options[0], options[1]])),
            TrainingSample(pdf_data_2, LabeledData(values=[options[1]])),
            TrainingSample(pdf_data_3, LabeledData(values=[options[2], options[0]])),
            TrainingSample(pdf_data_4, LabeledData(values=[options[1], options[2]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        fast_segment_selector_fuzzy_commas = FastSegmentSelectorFuzzyCommas()
        performance = fast_segment_selector_fuzzy_commas.get_performance(multi_option_data)

        self.assertEqual(100, performance)

    def test_performance_83(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_1 = PdfData.from_texts(["1, 2"])
        pdf_data_2 = PdfData.from_texts(["2"])
        pdf_data_3 = PdfData.from_texts(["3, 1"])
        pdf_data_4 = PdfData.from_texts(["4"])

        samples = [
            TrainingSample(pdf_data_1, LabeledData(values=[options[0], options[1]])),
            TrainingSample(pdf_data_2, LabeledData(values=[options[1]])),
            TrainingSample(pdf_data_3, LabeledData(values=[options[2], options[0]])),
            TrainingSample(pdf_data_4, LabeledData(values=[options[1], options[2]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        fast_segment_selector_fuzzy_commas = FastSegmentSelectorFuzzyCommas()
        performance = fast_segment_selector_fuzzy_commas.get_performance(multi_option_data)

        self.assertAlmostEqual(83.33333333333334, performance)

    def test_predictions(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [
            Option(id="1", label="1"),
            Option(id="2", label="2"),
            Option(id="3", label="3"),
            Option(id="4", label="4"),
        ]

        pdf_data_1 = PdfData.from_texts(["1, 2"])
        pdf_data_2 = PdfData.from_texts(["2"])
        pdf_data_3 = PdfData.from_texts(["3, 1"])
        pdf_data_4 = PdfData.from_texts(["2, 3"])
        pdf_data_5 = PdfData.from_texts(["4, 3"])

        samples = [
            TrainingSample(pdf_data_1, LabeledData(values=[options[0], options[1]])),
            TrainingSample(pdf_data_2, LabeledData(values=[options[1]])),
            TrainingSample(pdf_data_3, LabeledData(values=[options[2], options[0]])),
            TrainingSample(pdf_data_4, LabeledData(values=[options[1], options[2]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        fast_segment_selector_fuzzy_commas = FastSegmentSelectorFuzzyCommas()
        fast_segment_selector_fuzzy_commas.train(multi_option_data)

        prediction_samples = [
            TrainingSample(pdf_data_1, LabeledData(values=[])),
            TrainingSample(pdf_data_5, LabeledData(values=[])),
        ]
        prediction_multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=prediction_samples, extraction_identifier=extraction_identifier
        )
        predictions = fast_segment_selector_fuzzy_commas.predict(prediction_multi_option_data)

        self.assertEqual(2, len(predictions))

        self.assertEqual(2, len(predictions[0]))
        self.assertTrue(Option(id="1", label="1") in predictions[0])
        self.assertTrue(Option(id="2", label="2") in predictions[0])

        self.assertEqual(2, len(predictions[1]))
        self.assertTrue(Option(id="4", label="4") in predictions[1])
        self.assertTrue(Option(id="3", label="3") in predictions[1])

    def test_predictions_when_empy_data(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_1 = PdfData.from_texts(["1, 2"])
        pdf_data_2 = PdfData.from_texts(["2"])
        pdf_data_3 = PdfData.from_texts(["3, 1"])

        samples = [
            TrainingSample(pdf_data_1, LabeledData(values=[options[0], options[1]])),
            TrainingSample(pdf_data_2, LabeledData(values=[options[1]])),
            TrainingSample(pdf_data_3, LabeledData(values=[options[2], options[0]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        fast_segment_selector_fuzzy_commas = FastSegmentSelectorFuzzyCommas()
        fast_segment_selector_fuzzy_commas.train(multi_option_data)

        prediction_samples = [
            TrainingSample(pdf_data_1, LabeledData(values=[])),
            TrainingSample(PdfData(None), LabeledData(values=[])),
            TrainingSample(PdfData.from_texts([]), LabeledData(values=[])),
            TrainingSample(PdfData.from_texts([""]), LabeledData(values=[])),
        ]
        prediction_multi_option_data = ExtractionData(
            multi_value=True, options=options, samples=prediction_samples, extraction_identifier=extraction_identifier
        )
        predictions = fast_segment_selector_fuzzy_commas.predict(prediction_multi_option_data)

        self.assertEqual(4, len(predictions))

        self.assertEqual(2, len(predictions[0]))
        self.assertTrue(Option(id="1", label="1") in predictions[0])
        self.assertTrue(Option(id="2", label="2") in predictions[0])

        self.assertEqual(0, len(predictions[1]))
        self.assertEqual(0, len(predictions[2]))
        self.assertEqual(0, len(predictions[3]))