import json
import os
import shutil
from os.path import exists, join

from unittest import TestCase

import mongomock
import pymongo
from pdf_token_type_labels.TokenType import TokenType

from config import DATA_PATH, APP_PATH, MONGO_HOST, MONGO_PORT
from data.ExtractionTask import ExtractionTask
from data.Option import Option
from data.Params import Params
from data.SegmentBox import SegmentBox
from data.Suggestion import Suggestion
from Extractor import Extractor


class TestMetadataExtractor(TestCase):
    test_xml_path = f"{APP_PATH}/tenant_test/extraction_id/xml_to_train/test.xml"
    model_path = f"{APP_PATH}/tenant_test/extraction_id/segment_predictor_model/model.model"

    @mongomock.patch(servers=[f"{MONGO_HOST}:{MONGO_PORT}"])
    def test_create_model_error_when_blank_document(self):
        tenant = "segment_test"
        extraction_id = "extraction_id"

        base_path = join(DATA_PATH, tenant, extraction_id)

        mongo_client = pymongo.MongoClient(f"{MONGO_HOST}:{MONGO_PORT}")

        json_data = {
            "tenant": tenant,
            "id": extraction_id,
            "xml_file_name": "blank.xml",
            "label_text": "text",
            "language_iso": "en",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
            "label_segments_boxes": [{"left": 123, "top": 48, "width": 83, "height": 12, "page_number": 1, "type": "TEXT"}],
        }
        mongo_client.pdf_metadata_extraction.labeled_data.insert_one(json_data)

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

        os.makedirs(f"{base_path}/xml_to_train")
        shutil.copy(self.test_xml_path, f"{base_path}/xml_to_train/test.xml")

        task = ExtractionTask(
            tenant=tenant,
            task=Extractor.CREATE_MODEL_TASK_NAME,
            params=Params(id=extraction_id),
        )
        task_calculated, error = Extractor.calculate_task(task)

        self.assertFalse(task_calculated)

        shutil.rmtree(join(DATA_PATH, tenant))

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_create_model_error_when_no_files(self):
        tenant = "error_segment_test"
        extraction_id = "error_extraction_id"

        task = ExtractionTask(
            tenant=tenant,
            task=Extractor.CREATE_MODEL_TASK_NAME,
            params=Params(id=extraction_id),
        )
        task_calculated, error = Extractor.calculate_task(task)

        self.assertFalse(task_calculated)
        self.assertEqual(error, "No labeled data to create model")

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_create_model_should_do_nothing_when_no_xml(self):
        tenant = "segment_test"
        extraction_id = "extraction_id"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")
        json_data = {
            "tenant": tenant,
            "id": extraction_id,
            "xml_file_name": "not_found.xml_to_train",
            "language_iso": "en",
            "label_text": "text",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
            "label_segments_boxes": [{"left": 125, "top": 247, "width": 319, "height": 29, "page_number": 1}],
        }

        mongo_client.pdf_metadata_extraction.labeled_data.insert_one(json_data)

        task = ExtractionTask(
            tenant=tenant,
            task=Extractor.CREATE_MODEL_TASK_NAME,
            params=Params(id=extraction_id),
        )
        Extractor.calculate_task(task)

        self.assertFalse(os.path.exists(f"{DATA_PATH}/segment_test/extraction_id/xml_to_train"))
        self.assertFalse(os.path.exists(f"{DATA_PATH}/segment_test/extraction_id/segment_predictor_model/model.model"))

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_calculate_suggestions(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "segment_test"
        extraction_id = "extraction_id"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{APP_PATH}/tenant_test", f"{DATA_PATH}/{tenant}")

        labeled_data_json = {
            "tenant": tenant,
            "id": extraction_id,
            "xml_file_name": "test.xml",
            "language_iso": "en",
            "label_text": "Original: English",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
            "label_segments_boxes": [
                json.loads(
                    SegmentBox(left=400, top=115, width=74, height=9, page_number=1, type=TokenType.TEXT).model_dump_json()
                )
            ],
        }

        mongo_client.pdf_metadata_extraction.labeled_data.insert_one(labeled_data_json)

        Extractor.calculate_task(
            ExtractionTask(
                tenant=tenant,
                task=Extractor.CREATE_MODEL_TASK_NAME,
                params=Params(id=extraction_id),
            )
        )

        to_predict_json = {
            "tenant": tenant,
            "id": extraction_id,
            "xml_file_name": "test.xml",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
        }

        mongo_client.pdf_metadata_extraction.prediction_data.insert_one(to_predict_json)

        task = ExtractionTask(
            tenant=tenant,
            task=Extractor.SUGGESTIONS_TASK_NAME,
            params=Params(id=extraction_id),
        )
        task_calculated, error = Extractor.calculate_task(task)

        documents_count = mongo_client.pdf_metadata_extraction.suggestions.count_documents({})
        suggestion = Suggestion(**mongo_client.pdf_metadata_extraction.suggestions.find_one())

        self.assertTrue(task_calculated)
        self.assertEqual(1, documents_count)

        self.assertEqual(tenant, suggestion.tenant)
        self.assertEqual(extraction_id, suggestion.id)
        self.assertEqual("test.xml", suggestion.xml_file_name)
        self.assertEqual("Original: English", suggestion.segment_text)
        self.assertEqual("Original: English", suggestion.text)
        self.assertEqual(1, suggestion.page_number)

        self.assertEqual(len(suggestion.segments_boxes), 1)
        self.assertEqual(397.0, suggestion.segments_boxes[0].left)
        self.assertEqual(114.0, suggestion.segments_boxes[0].top)
        self.assertEqual(77.0, suggestion.segments_boxes[0].width)
        self.assertEqual(11.0, suggestion.segments_boxes[0].height)
        self.assertEqual(1, suggestion.segments_boxes[0].page_number)

        self.assertIsNone(mongo_client.pdf_metadata_extraction.prediction_data.find_one())

        self.assertTrue(os.path.exists(f"{DATA_PATH}/{tenant}/{extraction_id}/xml_to_predict/spanish.xml"))
        self.assertFalse(os.path.exists(f"{DATA_PATH}/{tenant}/{extraction_id}/xml_to_predict/test.xml"))

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions_page_2(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "segment_test"
        extraction_id = "extraction_id"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{APP_PATH}/tenant_test", f"{DATA_PATH}/{tenant}")

        labeled_data_json = {
            "tenant": tenant,
            "id": extraction_id,
            "xml_file_name": "test.xml",
            "language_iso": "en",
            "label_text": "text",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [
                json.loads(
                    SegmentBox(left=0, top=130, width=612, height=70, page_number=2, type=TokenType.TEXT).model_dump_json()
                )
            ],
            "label_segments_boxes": [
                json.loads(
                    SegmentBox(left=300, top=150, width=5, height=5, page_number=2, type=TokenType.TEXT).model_dump_json()
                )
            ],
        }

        mongo_client.pdf_metadata_extraction.labeled_data.insert_one(labeled_data_json)

        Extractor.calculate_task(
            ExtractionTask(
                tenant=tenant,
                task=Extractor.CREATE_MODEL_TASK_NAME,
                params=Params(id=extraction_id),
            )
        )

        to_predict_json = {
            "tenant": tenant,
            "id": extraction_id,
            "xml_file_name": "test.xml",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [
                json.loads(SegmentBox(left=0, top=130, width=612, height=70, page_number=2).model_dump_json())
            ],
        }

        mongo_client.pdf_metadata_extraction.prediction_data.insert_one(to_predict_json)

        task_calculated, error = Extractor.calculate_task(
            ExtractionTask(
                tenant=tenant,
                task=Extractor.SUGGESTIONS_TASK_NAME,
                params=Params(id=extraction_id),
            )
        )

        documents_count = mongo_client.pdf_metadata_extraction.suggestions.count_documents({})
        suggestion = Suggestion(**mongo_client.pdf_metadata_extraction.suggestions.find_one())

        self.assertTrue(task_calculated)
        self.assertEqual(1, documents_count)
        self.assertEqual(tenant, suggestion.tenant)
        self.assertEqual(extraction_id, suggestion.id)
        self.assertEqual("test.xml", suggestion.xml_file_name)
        self.assertTrue("In accordance with paragraph" in suggestion.segment_text)
        self.assertTrue("every four years" in suggestion.text)
        self.assertEqual(2, suggestion.page_number)
        self.assertTrue(os.path.exists(f"{DATA_PATH}/{tenant}/{extraction_id}/xml_to_predict"))
        self.assertFalse(os.path.exists(f"{DATA_PATH}/{tenant}/{extraction_id}/xml_to_predict/test.xml"))

        self.assertEqual(len(suggestion.segments_boxes), 1)
        self.assertEqual(134.0, suggestion.segments_boxes[0].left)
        self.assertEqual(126.0, suggestion.segments_boxes[0].top)
        self.assertEqual(357.0, suggestion.segments_boxes[0].width)
        self.assertEqual(72, suggestion.segments_boxes[0].height)
        self.assertEqual(2, suggestion.segments_boxes[0].page_number)

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_semantic_suggestions(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "tenant_to_be_removed"
        extraction_id = "extraction_id"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{APP_PATH}/tenant_test", f"{DATA_PATH}/{tenant}")

        to_predict_json = [
            {
                "xml_file_name": "test.xml",
                "id": extraction_id,
                "tenant": tenant,
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            },
            {
                "xml_file_name": "test.xml",
                "id": extraction_id,
                "tenant": tenant,
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            },
        ]

        mongo_client.pdf_metadata_extraction.prediction_data.insert_many(to_predict_json)

        for i in range(7):
            labeled_data_json = {
                "id": extraction_id,
                "tenant": tenant,
                "xml_file_name": "test.xml",
                "language_iso": "en",
                "label_text": "English",
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
                "label_segments_boxes": [SegmentBox(left=397, top=115, width=74, height=9, page_number=1).to_dict()],
            }

            mongo_client.pdf_metadata_extraction.labeled_data.insert_one(labeled_data_json)

        Extractor.calculate_task(
            ExtractionTask(
                tenant=tenant,
                task=Extractor.CREATE_MODEL_TASK_NAME,
                params=Params(id=extraction_id),
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
        find_filter = {"id": extraction_id, "tenant": tenant}
        for document in mongo_client.pdf_metadata_extraction.suggestions.find(find_filter):
            suggestions.append(Suggestion(**document))

        self.assertTrue(task_calculated)
        self.assertEqual(2, len(suggestions))
        self.assertEqual({tenant}, {x.tenant for x in suggestions})
        self.assertEqual({extraction_id}, {x.id for x in suggestions})
        self.assertEqual({"test.xml"}, {x.xml_file_name for x in suggestions})
        self.assertEqual({"Original: English"}, {x.segment_text for x in suggestions})
        self.assertEqual({"English"}, {x.text for x in suggestions})

        self.assertEqual({1}, set([len(x.segments_boxes) for x in suggestions]))
        self.assertEqual(397.0, suggestions[0].segments_boxes[0].left)
        self.assertEqual(114.0, suggestions[0].segments_boxes[0].top)
        self.assertEqual(77.0, suggestions[0].segments_boxes[0].width)
        self.assertEqual(11.0, suggestions[0].segments_boxes[0].height)
        self.assertEqual(1, suggestions[0].segments_boxes[0].page_number)

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_semantic_suggestions_numeric(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "tenant_to_be_removed"
        extraction_id = "extraction_id"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{APP_PATH}/tenant_test", f"{DATA_PATH}/{tenant}")

        to_predict_json = [
            {
                "xml_file_name": "test.xml",
                "id": extraction_id,
                "tenant": tenant,
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            }
        ]

        mongo_client.pdf_metadata_extraction.prediction_data.insert_many(to_predict_json)

        for i in range(7):
            labeled_data_json = {
                "id": extraction_id,
                "tenant": tenant,
                "xml_file_name": "test.xml",
                "language_iso": "en",
                "label_text": "15",
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
                params=Params(id=extraction_id),
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
        find_filter = {"id": extraction_id, "tenant": tenant}
        for document in mongo_client.pdf_metadata_extraction.suggestions.find(find_filter):
            suggestions.append(Suggestion(**document))

        self.assertTrue(task_calculated)
        self.assertEqual(1, len(suggestions))
        self.assertEqual({tenant}, {x.tenant for x in suggestions})
        self.assertEqual({extraction_id}, {x.id for x in suggestions})
        self.assertEqual({"test.xml"}, {x.xml_file_name for x in suggestions})
        self.assertEqual({"15 February 2021"}, {x.segment_text for x in suggestions})
        self.assertEqual({"15"}, {x.text for x in suggestions})

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_semantic_suggestions_spanish(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "tenant_to_be_removed"
        extraction_id = "spa"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{APP_PATH}/tenant_test", f"{DATA_PATH}/{tenant}")
        shutil.copytree(
            f"{DATA_PATH}/{tenant}/extraction_id/xml_to_train",
            f"{DATA_PATH}/{tenant}/{extraction_id}/xml_to_train",
        )
        shutil.copytree(
            f"{DATA_PATH}/{tenant}/extraction_id/xml_to_train",
            f"{DATA_PATH}/{tenant}/{extraction_id}/xml_to_predict",
        )

        samples_number = 20
        for i in range(samples_number):
            labeled_data_json = {
                "tenant": tenant,
                "id": extraction_id,
                "xml_file_name": "spanish.xml",
                "language_iso": "spa",
                "label_text": "día",
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
                "label_segments_boxes": [
                    json.loads(SegmentBox(left=289, top=206, width=34, height=10, page_number=1).model_dump_json())
                ],
            }

            mongo_client.pdf_metadata_extraction.labeled_data.insert_one(labeled_data_json)
        to_predict_json = [
            {
                "tenant": tenant,
                "id": extraction_id,
                "xml_file_name": "spanish.xml",
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            },
            {
                "id": extraction_id,
                "tenant": tenant,
                "xml_file_name": "spanish.xml",
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            },
        ]

        mongo_client.pdf_metadata_extraction.prediction_data.insert_many(to_predict_json)

        Extractor.calculate_task(
            ExtractionTask(
                tenant=tenant,
                task=Extractor.CREATE_MODEL_TASK_NAME,
                params=Params(id=extraction_id),
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
        find_filter = {"id": extraction_id, "tenant": tenant}
        for document in mongo_client.pdf_metadata_extraction.suggestions.find(find_filter):
            suggestions.append(Suggestion(**document))

        self.assertTrue(task_calculated)
        self.assertEqual(2, len(suggestions))
        self.assertEqual({tenant}, {x.tenant for x in suggestions})
        self.assertEqual({extraction_id}, {x.id for x in suggestions})
        self.assertEqual({"spanish.xml"}, {x.xml_file_name for x in suggestions})
        self.assertEqual({"por día"}, {x.segment_text for x in suggestions})
        self.assertEqual({"día"}, {x.text for x in suggestions})

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions_no_files_error(self):
        tenant = "error_segment_test"
        extraction_id = "error_extraction_id"

        base_path = join(DATA_PATH, tenant, extraction_id)

        if not exists(f"{base_path}/segment_predictor_model"):
            os.makedirs(f"{base_path}/segment_predictor_model")

        shutil.copy(self.model_path, f"{base_path}/segment_predictor_model/")

        task = ExtractionTask(
            tenant=tenant,
            task=Extractor.SUGGESTIONS_TASK_NAME,
            params=Params(id=extraction_id),
        )
        task_calculated, error = Extractor.calculate_task(task)

        self.assertFalse(task_calculated)
        self.assertEqual(error, "No data to calculate suggestions")

        shutil.rmtree(join(DATA_PATH, tenant))

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions_no_model_error(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "error_segment_test"
        extraction_id = "error_extraction_id"

        base_path = join(DATA_PATH, tenant, extraction_id)

        os.makedirs(f"{base_path}/xml_to_predict")
        shutil.copy(self.test_xml_path, f"{base_path}/xml_to_predict/test.xml")

        to_predict_json = [
            {
                "tenant": tenant,
                "id": extraction_id,
                "xml_file_name": "test.xml",
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            }
        ]

        mongo_client.pdf_metadata_extraction.prediction_data.insert_many(to_predict_json)

        task = ExtractionTask(
            tenant=tenant,
            task=Extractor.SUGGESTIONS_TASK_NAME,
            params=Params(id=extraction_id),
        )
        task_calculated, error = Extractor.calculate_task(task)

        self.assertFalse(task_calculated)
        self.assertEqual(error, "No model")

        shutil.rmtree(join(DATA_PATH, tenant))

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions_blank_document(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "segment_test"
        extraction_id = "extraction_id"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{APP_PATH}/tenant_test", f"{DATA_PATH}/{tenant}")
        shutil.rmtree(
            f"{DATA_PATH}/{tenant}/{extraction_id}/semantic_model",
            ignore_errors=True,
        )

        to_predict_json = {
            "tenant": tenant,
            "id": extraction_id,
            "xml_file_name": "blank.xml",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
        }

        mongo_client.pdf_metadata_extraction.prediction_data.insert_one(to_predict_json)

        task = ExtractionTask(
            tenant=tenant,
            task=Extractor.CREATE_MODEL_TASK_NAME,
            params=Params(id=extraction_id),
        )
        task_calculated, error = Extractor.calculate_task(task)

        self.assertFalse(task_calculated)

        self.assertIsNone(mongo_client.pdf_metadata_extraction.labeled_data.find_one({}))
        self.assertFalse(exists(join(DATA_PATH, tenant, extraction_id, "xml_to_train")))

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions_no_pages_document(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "segment_test"
        extraction_id = "extraction_id"

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{APP_PATH}/tenant_test", f"{DATA_PATH}/{tenant}")
        shutil.rmtree(
            f"{DATA_PATH}/{tenant}/{extraction_id}/semantic_model",
            ignore_errors=True,
        )

        to_predict_json = {
            "tenant": tenant,
            "id": extraction_id,
            "xml_file_name": "no_pages.xml",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
        }

        mongo_client.pdf_metadata_extraction.prediction_data.insert_one(to_predict_json)

        task = ExtractionTask(
            tenant=tenant,
            task=Extractor.CREATE_MODEL_TASK_NAME,
            params=Params(id=extraction_id),
        )
        task_calculated, error = Extractor.calculate_task(task)

        self.assertFalse(task_calculated)

        self.assertIsNone(mongo_client.pdf_metadata_extraction.labeled_data.find_one({}))
        self.assertFalse(exists(join(DATA_PATH, tenant, extraction_id, "xml_to_train")))

        shutil.rmtree(join(DATA_PATH, tenant), ignore_errors=True)

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
                "id": extraction_id,
                "tenant": tenant,
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            }
        ]

        mongo_client.pdf_metadata_extraction.prediction_data.insert_many(to_predict_json)

        options = [{"id": f"id{n}", "label": str(n)} for n in range(16)]
        for i in range(7):
            labeled_data_json = {
                "id": extraction_id,
                "tenant": tenant,
                "xml_file_name": "test.xml",
                "language_iso": "en",
                "options": [{"id": "id15", "label": "15"}],
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
        find_filter = {"id": extraction_id, "tenant": tenant}
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
