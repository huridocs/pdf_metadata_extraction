import shutil
from os.path import join
from unittest import TestCase

import torch

from config import DATA_PATH
from data.ExtractionData import ExtractionData
from data.ExtractionIdentifier import ExtractionIdentifier
from data.LabeledData import LabeledData
from data.Option import Option
from data.PdfData import PdfData
from data.TrainingSample import TrainingSample
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SingleLabelSetFitMultilingualMethod import (
    SingleLabelSetFitMultilingualMethod,
)


class TestSetFitSingleLabelMultilingualMethod(TestCase):
    TENANT = "single_label_setfit_multilingual_test"
    extraction_id = "extraction_id"

    def tearDown(self):
        shutil.rmtree(join(DATA_PATH, self.TENANT), ignore_errors=True)

    def test_train_and_predict(self):
        if not torch.cuda.is_available():
            return
        extraction_identifier = ExtractionIdentifier(run_name=self.TENANT, extraction_name=self.extraction_id)
        options = [Option(id="0", label="0"), Option(id="1", label="1"), Option(id="2", label="2")]

        pdf_data_1 = PdfData.from_texts(["point 1"])
        pdf_data_2 = PdfData.from_texts(["point 2"])
        pdf_data_3 = PdfData.from_texts(["point 3"])

        samples = [
            TrainingSample(pdf_data_1, LabeledData(values=[options[0]])),
            TrainingSample(pdf_data_2, LabeledData(values=[options[1]])),
            TrainingSample(pdf_data_3, LabeledData(values=[options[2]])),
        ]

        extraction_data = ExtractionData(
            multi_value=False, options=options, samples=samples, extraction_identifier=extraction_identifier
        )
        setfit_english_method = SingleLabelSetFitMultilingualMethod(extraction_identifier, options, False)

        try:
            setfit_english_method.train(extraction_data)
        except Exception as e:
            self.fail(f"train() raised {type(e).__name__}")

        prediction_sample_1 = TrainingSample(pdf_data=pdf_data_1)
        prediction_sample_2 = TrainingSample(pdf_data=pdf_data_2)
        prediction_sample_3 = TrainingSample(pdf_data=pdf_data_3)
        prediction_samples = [prediction_sample_1, prediction_sample_2, prediction_sample_3]

        prediction_data = ExtractionData(
            multi_value=False, options=options, samples=prediction_samples, extraction_identifier=extraction_identifier
        )
        predictions = setfit_english_method.predict(prediction_data)

        self.assertEqual(3, len(predictions))
        self.assertIn(Option(id="0", label="0"), predictions[0])
        self.assertIn(Option(id="1", label="1"), predictions[1])
        self.assertNotIn(Option(id="5", label="5"), predictions[0])
        self.assertNotIn(Option(id="4", label="4"), predictions[1])
