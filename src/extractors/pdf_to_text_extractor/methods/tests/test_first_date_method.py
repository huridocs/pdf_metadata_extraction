from unittest import TestCase

from data.ExtractionData import ExtractionData
from data.ExtractionIdentifier import ExtractionIdentifier
from data.LabeledData import LabeledData
from data.PredictionSample import PredictionSample
from data.TrainingSample import TrainingSample
from extractors.pdf_to_text_extractor.methods.FirstDateMethod import FirstDateMethod

extraction_identifier = ExtractionIdentifier(run_name="date_test", extraction_name="date_test")


class TestFirstDateMethod(TestCase):
    def test_predict_multiple_dates_spanish(self):
        text = "Informe no. 52/16 caso 12.521 fondo Maria Laura órdenes guerra y otros Chile 30 de noviembre de 2016"
        sample = TrainingSample(labeled_data=LabeledData(label_text="2016-11-30", language_iso="es"), tags_texts=[text])

        extraction_data = ExtractionData(samples=[sample for _ in range(6)], extraction_identifier=extraction_identifier)
        first_date_method = FirstDateMethod(extraction_identifier)

        first_date_method.train(extraction_data)

        predictions = first_date_method.predict([PredictionSample.from_text(text)])
        self.assertEqual(["2016-11-30"], predictions)