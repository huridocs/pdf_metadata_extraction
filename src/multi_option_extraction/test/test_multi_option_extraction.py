import json
import shutil
from os.path import join, exists
from unittest import TestCase

import mongomock
import pymongo

from Extractor import Extractor
from config import DATA_PATH, APP_PATH
from data.ExtractionIdentifier import ExtractionIdentifier
from data.ExtractionTask import ExtractionTask
from data.Option import Option
from data.Params import Params
from data.PdfTagData import PdfTagData
from data.SegmentBox import SegmentBox
from data.SemanticPredictionData import SemanticPredictionData
from data.Suggestion import Suggestion
from metadata_extraction.PdfData import PdfData
from metadata_extraction.PdfDataSegment import PdfDataSegment
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
            multi_value=False, options=options, samples=samples, extraction_identifier=extraction_identifier
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
        multi_option_extraction = MultiOptionExtractor(self.TENANT, self.extraction_id)
        multi_option_extraction.METHODS = [TfIdfMethod]
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_segment_1 = PdfTagData.from_texts(["point one point two"])
        pdf_data_segment_2 = PdfTagData.from_texts(["point two"])
        pdf_data_segment_3 = PdfTagData.from_texts(["point three point one"])

        samples = [
            MultiOptionSample(pdf_data_segment_1, [options[0], options[1]], "en"),
            MultiOptionSample(pdf_data_segment_2, [options[1]], "en"),
            MultiOptionSample(pdf_data_segment_3, [options[2], options[0]], "en"),
        ]

        multi_option_data = MultiOptionData(multi_value=True, options=options, samples=samples)
        multi_option_extraction.create_model(multi_option_data)

        semantic_prediction_data_1 = SemanticPredictionData(pdf_tags_data=pdf_data_segment_1)
        semantic_prediction_data_3 = SemanticPredictionData(pdf_tags_data=pdf_data_segment_3)
        semantic_predictions_data = [semantic_prediction_data_1, semantic_prediction_data_3]
        predictions = multi_option_extraction.get_multi_option_predictions(semantic_predictions_data)

        self.assertEqual(2, len(predictions))
        self.assertEqual([options[0], options[1]], sorted(predictions[0].values, key=lambda x: x.id))
        self.assertEqual([options[0], options[2]], sorted(predictions[1].values, key=lambda x: x.id))

    def test_bert(self):
        multi_option_extraction = MultiOptionExtractor(self.TENANT, self.extraction_id)
        multi_option_extraction.METHODS = [BertSeqSteps]
        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        pdf_data_segment_1 = PdfTagData.from_texts(["point one point two"])
        pdf_data_segment_2 = PdfTagData.from_texts(["point two"])
        pdf_data_segment_3 = PdfTagData.from_texts(["point three point one"])

        samples = [
            MultiOptionSample(pdf_data_segment_1, [options[0], options[1]], "en"),
            MultiOptionSample(pdf_data_segment_2, [options[1]], "en"),
            MultiOptionSample(pdf_data_segment_3, [options[2], options[0]], "en"),
        ]

        multi_option_data = MultiOptionData(multi_value=True, options=options, samples=samples)
        multi_option_extraction.create_model(multi_option_data)

        semantic_prediction_data_1 = SemanticPredictionData(pdf_tags_data=pdf_data_segment_1)
        semantic_prediction_data_3 = SemanticPredictionData(pdf_tags_data=pdf_data_segment_3)
        semantic_predictions_data = [semantic_prediction_data_1, semantic_prediction_data_3]
        predictions = multi_option_extraction.get_multi_option_predictions(semantic_predictions_data)

        self.assertEqual(2, len(predictions))
        self.assertEqual([options[0], options[1]], sorted(predictions[0].values, key=lambda x: x.id))
        self.assertEqual([options[0], options[2]], sorted(predictions[1].values, key=lambda x: x.id))

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_multi_option_suggestions(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "tenant_to_be_removed"
        extraction_id = "extraction_id"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{APP_PATH}/tenant_test", f"{DATA_PATH}/{tenant}")

        to_predict_json = [
            {
                "xml_file_name": "test.xml",
                "extraction_name": extraction_id,
                "run_name": tenant,
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            }
        ]

        mongo_client.pdf_metadata_extraction.prediction_data.insert_many(to_predict_json)

        options = [{"extraction_name": f"id{n}", "label": str(n)} for n in range(16)]
        for i in range(7):
            labeled_data_json = {
                "extraction_name": extraction_id,
                "run_name": tenant,
                "xml_file_name": "test.xml",
                "language_iso": "en",
                "options": [{"extraction_name": "id15", "label": "15"}],
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
                "label_segments_boxes": [
                    json.loads(SegmentBox(left=397, top=91, width=10, height=9, page_number=1).model_dump_json())
                ],
            }

            mongo_client.pdf_metadata_extraction.labeled_data.insert_one(labeled_data_json)

        Extractor.calculate_task(
            ExtractionTask(
                tenant=tenant,
                task=Extractor.CREATE_MODEL_TASK_NAME,
                params=Params(id=extraction_id, options=options, multi_value=False),
            )
        )

        task_calculated, error = Extractor.calculate_task(
            ExtractionTask(
                tenant=tenant,
                task=Extractor.SUGGESTIONS_TASK_NAME,
                params=Params(id=extraction_id),
            )
        )

        suggestions: list[Suggestion] = list()
        find_filter = {"extraction_name": extraction_id, "run_name": tenant}
        for document in mongo_client.pdf_metadata_extraction.suggestions.find(find_filter):
            suggestions.append(Suggestion(**document))

        self.assertTrue(task_calculated)
        self.assertEqual(1, len(suggestions))
        self.assertEqual(tenant, suggestions[0].tenant)
        self.assertEqual(extraction_id, suggestions[0].id)
        self.assertEqual("test.xml", suggestions[0].xml_file_name)
        self.assertEqual("15 February 2021", suggestions[0].segment_text)
        self.assertEqual([Option(id="id15", label="15")], suggestions[0].options)

        self.assertIsNone(mongo_client.pdf_metadata_extraction.labeled_data.find_one({}))
        self.assertFalse(exists(join(DATA_PATH, tenant, extraction_id, "xml_to_train")))

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
