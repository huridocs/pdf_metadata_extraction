from unittest import TestCase

from data.Option import Option
from data.PdfTagData import PdfTagData
from data.SemanticPredictionData import SemanticPredictionData
from multi_option_extraction.MultiOptionExtractionData import MultiOptionExtractionData, MultiOptionExtractionSample
from multi_option_extraction.MultiOptionExtractor import MultiOptionExtractor


class TestMultiOptionExtraction(TestCase):
    TENANT = "multi_option_extraction_test"
    extraction_id = "extraction_id"

    def test_single_value(self):
        multi_option_extraction = MultiOptionExtractor(self.TENANT, self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_tag_data_1 = PdfTagData.from_texts(["point 1"])
        pdf_tag_data_2 = PdfTagData.from_texts(["point 2"])
        pdf_tag_data_3 = PdfTagData.from_texts(["point 3"])

        samples = [
            MultiOptionExtractionSample(pdf_tag_data_1, [options[0]], "en"),
            MultiOptionExtractionSample(pdf_tag_data_2, [options[1]], "en"),
            MultiOptionExtractionSample(pdf_tag_data_3, [options[2]], "en"),
        ]

        multi_option_extraction_data = MultiOptionExtractionData(multi_value=False, options=options, samples=samples)
        multi_option_extraction.create_model(multi_option_extraction_data)

        semantic_prediction_data_1 = SemanticPredictionData(pdf_tags=pdf_tag_data_1)
        semantic_prediction_data_3 = SemanticPredictionData(pdf_tags=pdf_tag_data_3)
        semantic_predictions_data = [semantic_prediction_data_1, semantic_prediction_data_3]
        predictions = multi_option_extraction.get_multi_option_predictions(semantic_predictions_data)

        self.assertEqual(2, len(predictions))
        self.assertEqual([Option(id="1", label="1")], predictions[0].options)
        self.assertEqual([Option(id="3", label="3")], predictions[1].options)
