from unittest import TestCase

from data.Option import Option
from data.PdfTagData import PdfTagData
from data.SemanticPredictionData import SemanticPredictionData
from multi_option_extraction.MultiOptionExtractionData import MultiOptionExtractionData, MultiOptionExtractionSample
from multi_option_extraction.MultiOptionExtractor import MultiOptionExtractor
from multi_option_extraction.methods.BertBaseMethod import BertBaseMethod
from multi_option_extraction.methods.SetFitMethod import SetFitMethod
from multi_option_extraction.methods.TfIdfMethod import TfIdfMethod


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

    def test_multi_value(self):
        multi_option_extraction = MultiOptionExtractor(self.TENANT, self.extraction_id)
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_tag_data_1 = PdfTagData.from_texts(["point 1 point 2"])
        pdf_tag_data_2 = PdfTagData.from_texts(["point 2"])
        pdf_tag_data_3 = PdfTagData.from_texts(["point 3 point 1"])

        samples = [
            MultiOptionExtractionSample(pdf_tag_data_1, [options[0], options[1]], "en"),
            MultiOptionExtractionSample(pdf_tag_data_2, [options[1]], "en"),
            MultiOptionExtractionSample(pdf_tag_data_3, [options[2], options[0]], "en"),
        ]

        multi_option_extraction_data = MultiOptionExtractionData(multi_value=True, options=options, samples=samples)
        multi_option_extraction.create_model(multi_option_extraction_data)

        semantic_prediction_data_1 = SemanticPredictionData(pdf_tags=pdf_tag_data_1)
        semantic_prediction_data_3 = SemanticPredictionData(pdf_tags=pdf_tag_data_3)
        semantic_predictions_data = [semantic_prediction_data_1, semantic_prediction_data_3]
        predictions = multi_option_extraction.get_multi_option_predictions(semantic_predictions_data)

        self.assertEqual(2, len(predictions))
        self.assertEqual([options[0], options[1]], sorted(predictions[0].options, key=lambda x: x.id))
        self.assertEqual([options[0], options[2]], sorted(predictions[1].options, key=lambda x: x.id))

    def test_tf_idf(self):
        multi_option_extraction = MultiOptionExtractor(self.TENANT, self.extraction_id)
        multi_option_extraction.METHODS = [TfIdfMethod]
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_tag_data_1 = PdfTagData.from_texts(["point one point two"])
        pdf_tag_data_2 = PdfTagData.from_texts(["point two"])
        pdf_tag_data_3 = PdfTagData.from_texts(["point three point one"])

        samples = [
            MultiOptionExtractionSample(pdf_tag_data_1, [options[0], options[1]], "en"),
            MultiOptionExtractionSample(pdf_tag_data_2, [options[1]], "en"),
            MultiOptionExtractionSample(pdf_tag_data_3, [options[2], options[0]], "en"),
        ]

        multi_option_extraction_data = MultiOptionExtractionData(multi_value=True, options=options, samples=samples)
        multi_option_extraction.create_model(multi_option_extraction_data)

        semantic_prediction_data_1 = SemanticPredictionData(pdf_tags=pdf_tag_data_1)
        semantic_prediction_data_3 = SemanticPredictionData(pdf_tags=pdf_tag_data_3)
        semantic_predictions_data = [semantic_prediction_data_1, semantic_prediction_data_3]
        predictions = multi_option_extraction.get_multi_option_predictions(semantic_predictions_data)

        self.assertEqual(2, len(predictions))
        self.assertEqual([options[0], options[1]], sorted(predictions[0].options, key=lambda x: x.id))
        self.assertEqual([options[0], options[2]], sorted(predictions[1].options, key=lambda x: x.id))

    def test_setfit(self):
        multi_option_extraction = MultiOptionExtractor(self.TENANT, self.extraction_id)
        multi_option_extraction.METHODS = [SetFitMethod]
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_tag_data_1 = PdfTagData.from_texts(["point one point two"])
        pdf_tag_data_2 = PdfTagData.from_texts(["point two"])
        pdf_tag_data_3 = PdfTagData.from_texts(["point three point one"])

        samples = [
            MultiOptionExtractionSample(pdf_tag_data_1, [options[0], options[1]], "en"),
            MultiOptionExtractionSample(pdf_tag_data_2, [options[1]], "en"),
            MultiOptionExtractionSample(pdf_tag_data_3, [options[2], options[0]], "en"),
        ]

        multi_option_extraction_data = MultiOptionExtractionData(multi_value=True, options=options, samples=samples)
        multi_option_extraction.create_model(multi_option_extraction_data)

        semantic_prediction_data_1 = SemanticPredictionData(pdf_tags=pdf_tag_data_1)
        semantic_prediction_data_3 = SemanticPredictionData(pdf_tags=pdf_tag_data_3)
        semantic_predictions_data = [semantic_prediction_data_1, semantic_prediction_data_3]
        predictions = multi_option_extraction.get_multi_option_predictions(semantic_predictions_data)

        self.assertEqual(2, len(predictions))
        self.assertEqual([options[0], options[1]], sorted(predictions[0].options, key=lambda x: x.id))
        self.assertEqual([options[0], options[2]], sorted(predictions[1].options, key=lambda x: x.id))

    def test_bert(self):
        multi_option_extraction = MultiOptionExtractor(self.TENANT, self.extraction_id)
        multi_option_extraction.METHODS = [BertBaseMethod]
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_tag_data_1 = PdfTagData.from_texts(["point one point two"])
        pdf_tag_data_2 = PdfTagData.from_texts(["point two"])
        pdf_tag_data_3 = PdfTagData.from_texts(["point three point one"])

        samples = [
            MultiOptionExtractionSample(pdf_tag_data_1, [options[0], options[1]], "en"),
            MultiOptionExtractionSample(pdf_tag_data_2, [options[1]], "en"),
            MultiOptionExtractionSample(pdf_tag_data_3, [options[2], options[0]], "en"),
        ]

        multi_option_extraction_data = MultiOptionExtractionData(multi_value=True, options=options, samples=samples)
        multi_option_extraction.create_model(multi_option_extraction_data)

        semantic_prediction_data_1 = SemanticPredictionData(pdf_tags=pdf_tag_data_1)
        semantic_prediction_data_3 = SemanticPredictionData(pdf_tags=pdf_tag_data_3)
        semantic_predictions_data = [semantic_prediction_data_1, semantic_prediction_data_3]
        predictions = multi_option_extraction.get_multi_option_predictions(semantic_predictions_data)

        self.assertEqual(2, len(predictions))
        self.assertEqual([options[0], options[1]], sorted(predictions[0].options, key=lambda x: x.id))
        self.assertEqual([options[0], options[2]], sorted(predictions[1].options, key=lambda x: x.id))
