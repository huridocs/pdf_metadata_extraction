from unittest import TestCase
from data.ExtractionData import ExtractionData
from data.ExtractionIdentifier import ExtractionIdentifier
from data.LabeledData import LabeledData
from data.Option import Option
from data.PdfData import PdfData
from data.TrainingSample import TrainingSample
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SingleLabelSetFitMultilingualMethod import \
    SingleLabelSetFitMultilingualMethod


class TestSetFitSingleLabelMultilingualMethod(TestCase):
    TENANT = "multi_option_extraction_test"
    extraction_id = "extraction_id"

    def test_train_and_predict(self):
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3"), Option(id="4", label="4")]

        pdf_data_1 = PdfData.from_texts(["point 1"])
        pdf_data_2 = PdfData.from_texts(["point 2"])
        pdf_data_3 = PdfData.from_texts(["point 3"])
        pdf_data_4 = PdfData.from_texts(["point 4"])

        samples = [
            TrainingSample(pdf_data_1, LabeledData(values=[options[0]])),
            TrainingSample(pdf_data_2, LabeledData(values=[options[1]])),
            TrainingSample(pdf_data_3, LabeledData(values=[options[2]])),
            TrainingSample(pdf_data_4, LabeledData(values=[options[3]])),
        ]

        multi_option_data = ExtractionData(
            multi_value=False, options=options, samples=samples, extraction_identifier=extraction_identifier
        )

        setfit_english_method = SingleLabelSetFitMultilingualMethod(extraction_identifier, options, False)
        setfit_english_method.train(multi_option_data)

        prediction_sample_1 = TrainingSample(pdf_data=pdf_data_1)
        prediction_sample_3 = TrainingSample(pdf_data=pdf_data_3)
        prediction_samples = [prediction_sample_1, prediction_sample_3]

        prediction_data = ExtractionData(
            multi_value=False, options=options, samples=prediction_samples, extraction_identifier=extraction_identifier
        )

        predictions = setfit_english_method.predict(prediction_data)

        self.assertEqual(2, len(predictions))
        self.assertEqual([Option(id="1", label="1")], predictions[0])
        self.assertEqual([Option(id="3", label="3")], predictions[1])



