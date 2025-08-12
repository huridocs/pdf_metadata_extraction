import json
import os
import shutil
from os.path import join

import mongomock
import pymongo
from fastapi.testclient import TestClient
from unittest import TestCase

from pdf_token_type_labels.TokenType import TokenType
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.domain.SegmentBox import SegmentBox
from trainable_entity_extractor.domain.TrainingSample import TrainingSample

from drivers.rest.app import app
from config import MODELS_DATA_PATH, APP_PATH, MONGO_HOST, MONGO_PORT
from use_cases.ExtractorUseCase import ExtractorUseCase


class TestApp(TestCase):
    test_file_path = f"{APP_PATH}/tests/resources/tenant_test/extraction_id/xml_to_predict/test.xml"

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary test directory
        self.test_base_dir = os.path.join(MODELS_DATA_PATH, "test_temp")
        os.makedirs(self.test_base_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test environment after each test."""
        # Remove temporary test directory
        if os.path.exists(self.test_base_dir):
            shutil.rmtree(self.test_base_dir, ignore_errors=True)

    @staticmethod
    def _create_test_extraction_folder(extraction_identifier: ExtractionIdentifier):
        os.makedirs(extraction_identifier.get_path(), exist_ok=True)

        # Create some test files
        test_files = ["model.pkl", "training_data.json", "config.yaml", "logs.txt"]

        for filename in test_files:
            file_path = os.path.join(extraction_identifier.get_path(), filename)
            with open(file_path, "w") as f:
                f.write(f"Test content for {filename}")

    def test_info(self):
        with TestClient(app) as client:
            response = client.get("/")

        self.assertEqual(200, response.status_code)

    def test_post_train_xml_file(self):
        run_name = "endpoint_test"
        extraction_name = "extraction_id"

        shutil.rmtree(join(MODELS_DATA_PATH, run_name), ignore_errors=True)

        with open(self.test_file_path, "rb") as stream:
            files = {"file": stream}
            with TestClient(app) as client:
                response = client.post(f"/xml_to_train/{run_name}/{extraction_name}", files=files)

        self.assertEqual(200, response.status_code)
        to_train_xml_path = f"{MODELS_DATA_PATH}/{run_name}/{extraction_name}/xml_to_train/test.xml"
        self.assertTrue(os.path.exists(to_train_xml_path))

        shutil.rmtree(join(MODELS_DATA_PATH, run_name), ignore_errors=True)

    def test_post_xml_to_predict(self):
        tenant = "endpoint_test"
        extraction_id = "extraction_id"

        shutil.rmtree(join(MODELS_DATA_PATH, tenant), ignore_errors=True)

        with open(self.test_file_path, "rb") as stream:
            files = {"file": stream}
            with TestClient(app) as client:
                response = client.post(f"/xml_to_predict/{tenant}/{extraction_id}", files=files)

        self.assertEqual(200, response.status_code)
        to_train_xml_path = f"{MODELS_DATA_PATH}/{tenant}/{extraction_id}/xml_to_predict/test.xml"
        self.assertTrue(os.path.exists(to_train_xml_path))

        shutil.rmtree(join(MODELS_DATA_PATH, tenant), ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_post_labeled_data(self):
        tenant = "endpoint_test"
        extraction_id = "extraction_id"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        json_data = {
            "run_name": tenant,
            "extraction_name": extraction_id,
            "tenant": tenant,
            "id": extraction_id,
            "xml_file_name": "xml_file_name",
            "language_iso": "en",
            "label_text": "text",
            "page_width": 1.1,
            "page_height": 2.1,
            "xml_segments_boxes": [
                {"left": 1, "top": 2, "width": 3, "height": 4, "page_width": 5, "page_height": 6, "page_number": 5}
            ],
            "label_segments_boxes": [
                {"left": 8, "top": 12, "width": 16, "height": 20, "page_width": 5, "page_height": 6, "page_number": 10}
            ],
        }

        with TestClient(app) as client:
            response = client.post("/labeled_data", json=json_data)

        labeled_data_document = mongo_client.pdf_metadata_extraction.labeled_data.find_one()

        self.assertEqual(200, response.status_code)
        self.assertEqual(tenant, labeled_data_document["tenant"])
        self.assertEqual(extraction_id, labeled_data_document["id"])
        self.assertEqual("text", labeled_data_document["label_text"])
        self.assertEqual("en", labeled_data_document["language_iso"])
        self.assertEqual(1.1, labeled_data_document["page_width"])
        self.assertEqual(2.1, labeled_data_document["page_height"])
        self.assertEqual("xml_file_name", labeled_data_document["xml_file_name"])
        self.assertEqual(
            [
                {
                    "height": 4.0,
                    "left": 1.0,
                    "page_number": 5,
                    "top": 2.0,
                    "width": 3.0,
                    "page_width": 5,
                    "page_height": 6,
                    "segment_type": "Text",
                }
            ],
            labeled_data_document["xml_segments_boxes"],
        )
        self.assertEqual(
            [
                {
                    "height": 15,
                    "left": 6,
                    "page_number": 10,
                    "top": 9,
                    "width": 12,
                    "page_width": 5,
                    "page_height": 6,
                    "segment_type": "Text",
                }
            ],
            labeled_data_document["label_segments_boxes"],
        )

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_post_labeled_data_different_values(self):
        tenant = "different_endpoint_test"
        extraction_id = "different_extraction_id"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        json_data = {
            "tenant": tenant,
            "id": extraction_id,
            "xml_file_name": "different_xml_file_name",
            "language_iso": "spa",
            "label_text": "other_text",
            "page_width": 3.1,
            "page_height": 4.1,
            "xml_segments_boxes": [],
            "label_segments_boxes": [],
        }
        with TestClient(app) as client:
            response = client.post("/labeled_data", json=json_data)

        labeled_data_document = mongo_client.pdf_metadata_extraction.labeled_data.find_one()

        self.assertEqual(200, response.status_code)
        self.assertEqual(tenant, labeled_data_document["tenant"])
        self.assertEqual(extraction_id, labeled_data_document["id"])
        self.assertEqual("other_text", labeled_data_document["label_text"])
        self.assertEqual("spa", labeled_data_document["language_iso"])
        self.assertEqual(3.1, labeled_data_document["page_width"])
        self.assertEqual(4.1, labeled_data_document["page_height"])
        self.assertEqual("different_xml_file_name", labeled_data_document["xml_file_name"])
        self.assertEqual([], labeled_data_document["xml_segments_boxes"])
        self.assertEqual([], labeled_data_document["label_segments_boxes"])

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_post_labeled_data_multi_option(self):
        tenant = "endpoint_test"
        extraction_id = "extraction_id"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        options_json = [{"id": "id1", "label": "label1"}, {"id": "id2", "label": "label2"}]

        json_data = {
            "tenant": tenant,
            "id": extraction_id,
            "xml_file_name": "xml_file_name",
            "language_iso": "en",
            "values": options_json,
            "page_width": 1.1,
            "page_height": 2.1,
            "xml_segments_boxes": [
                {"left": 1, "top": 2, "width": 3, "height": 4, "page_width": 5, "page_height": 6, "page_number": 5}
            ],
            "label_segments_boxes": [
                {"left": 8, "top": 12, "width": 16, "height": 20, "page_width": 5, "page_height": 6, "page_number": 10}
            ],
        }

        with TestClient(app) as client:
            response = client.post("/labeled_data", json=json_data)

        labeled_data_document = mongo_client.pdf_metadata_extraction.labeled_data.find_one()

        self.assertEqual(200, response.status_code)
        self.assertEqual(tenant, labeled_data_document["tenant"])
        self.assertEqual(extraction_id, labeled_data_document["id"])
        self.assertEqual(options_json, labeled_data_document["values"])
        self.assertEqual("en", labeled_data_document["language_iso"])
        self.assertEqual(1.1, labeled_data_document["page_width"])
        self.assertEqual(2.1, labeled_data_document["page_height"])
        self.assertEqual("xml_file_name", labeled_data_document["xml_file_name"])
        self.assertEqual(
            [
                {
                    "height": 4.0,
                    "left": 1.0,
                    "page_width": 5,
                    "page_height": 6,
                    "page_number": 5,
                    "top": 2.0,
                    "width": 3.0,
                    "segment_type": "Text",
                }
            ],
            labeled_data_document["xml_segments_boxes"],
        )
        self.assertEqual(
            [
                {
                    "height": 15,
                    "left": 6,
                    "page_width": 5,
                    "page_height": 6,
                    "page_number": 10,
                    "top": 9,
                    "width": 12,
                    "segment_type": "Text",
                }
            ],
            labeled_data_document["label_segments_boxes"],
        )

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_post_prediction_data(self):
        tenant = "endpoint_test"
        extraction_id = "extraction_id"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        json_data = {
            "tenant": tenant,
            "id": extraction_id,
            "xml_file_name": "xml_file_name",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [
                {
                    "left": 6,
                    "top": 7,
                    "width": 8,
                    "height": 9,
                    "page_width": 5,
                    "page_height": 6,
                    "page_number": 10,
                    "segment_type": "Footnote",
                }
            ],
        }

        with TestClient(app) as client:
            response = client.post("/prediction_data", json=json_data)

        prediction_data_document = mongo_client.pdf_metadata_extraction.prediction_data.find_one()

        self.assertEqual(200, response.status_code)
        self.assertEqual(tenant, prediction_data_document["tenant"])
        self.assertEqual(extraction_id, prediction_data_document["id"])
        self.assertEqual(612, prediction_data_document["page_width"])
        self.assertEqual(792, prediction_data_document["page_height"])
        self.assertEqual("xml_file_name", prediction_data_document["xml_file_name"])
        self.assertEqual(
            [
                {
                    "left": 6,
                    "top": 7,
                    "width": 8,
                    "height": 9,
                    "page_width": 5,
                    "page_height": 6,
                    "page_number": 10,
                    "segment_type": "Footnote",
                }
            ],
            prediction_data_document["xml_segments_boxes"],
        )

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions(self):
        print(f"mongodb://{MONGO_HOST}:{MONGO_PORT}")
        tenant = "example_tenant_name"
        extraction_id = "prediction_extraction_id"

        mongo_client = pymongo.MongoClient(f"{MONGO_HOST}:{MONGO_PORT}")

        json_data = [
            {
                "run_name": "wrong tenant",
                "extraction_name": extraction_id,
                "tenant": "wrong tenant",
                "id": extraction_id,
                "xml_file_name": "one_file_name",
                "text": "one_text_predicted",
                "segment_text": "one_segment_text",
                "page_number": 1,
                "segments_boxes": [
                    {"left": 1, "top": 2, "width": 3, "height": 4, "page_width": 5, "page_height": 6, "page_number": 1}
                ],
            },
            {
                "run_name": tenant,
                "extraction_name": extraction_id,
                "tenant": tenant,
                "id": extraction_id,
                "xml_file_name": "one_file_name",
                "text": "one_text_predicted",
                "segment_text": "one_segment_text",
                "page_number": 2,
                "segments_boxes": [
                    {"left": 3, "top": 6, "width": 9, "height": 12, "page_width": 5, "page_height": 6, "page_number": 2}
                ],
            },
            {
                "run_name": tenant,
                "extraction_name": extraction_id,
                "tenant": tenant,
                "id": extraction_id,
                "xml_file_name": "other_file_name",
                "text": "other_text_predicted",
                "segment_text": "other_segment_text",
                "page_number": 3,
                "segments_boxes": [
                    {"left": 1, "top": 2, "width": 3, "height": 4, "page_width": 5, "page_height": 6, "page_number": 3}
                ],
            },
            {
                "run_name": tenant,
                "extraction_name": "wrong extraction name",
                "tenant": tenant,
                "id": "wrong extraction name",
                "xml_file_name": "other_file_name",
                "text": "other_text_predicted",
                "segment_text": "other_segment_text",
                "page_number": 4,
                "segments_boxes": [
                    {"left": 1, "top": 2, "width": 3, "height": 4, "page_width": 5, "page_height": 6, "page_number": 4}
                ],
            },
        ]

        mongo_client.pdf_metadata_extraction.suggestions.insert_many(json_data)

        with TestClient(app) as client:
            response = client.get(f"/get_suggestions/{tenant}/{extraction_id}")

        suggestions = json.loads(response.json())

        self.assertEqual(200, response.status_code)
        self.assertEqual(2, len(suggestions))

        self.assertEqual({tenant}, {x["tenant"] for x in suggestions})
        self.assertEqual({extraction_id}, {x["id"] for x in suggestions})

        self.assertEqual("one_file_name", suggestions[0]["xml_file_name"])
        self.assertEqual("one_segment_text", suggestions[0]["segment_text"])
        self.assertEqual("one_text_predicted", suggestions[0]["text"])
        self.assertEqual(2, suggestions[0]["page_number"])
        self.assertEqual(4, suggestions[0]["segments_boxes"][0]["left"])
        self.assertEqual(8, suggestions[0]["segments_boxes"][0]["top"])
        self.assertEqual(12, suggestions[0]["segments_boxes"][0]["width"])
        self.assertEqual(16, suggestions[0]["segments_boxes"][0]["height"])

        self.assertEqual("other_file_name", suggestions[1]["xml_file_name"])
        self.assertEqual("other_segment_text", suggestions[1]["segment_text"])
        self.assertEqual("other_text_predicted", suggestions[1]["text"])
        self.assertEqual(3, suggestions[1]["page_number"])

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions_multi_option(self):
        tenant = "example_tenant_name"
        extraction_id = "prediction_extraction_id"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        json_data = [
            {
                "run_name": "wrong tenant",
                "extraction_name": extraction_id,
                "tenant": "wrong tenant",
                "id": extraction_id,
                "xml_file_name": "one_file_name",
                "values": [{"id": "one_id", "label": "one_label", "segment_text": "one_segment_text"}],
                "segment_text": "one_segment_text",
                "page_number": 1,
                "segments_boxes": [
                    {"left": 1, "top": 2, "width": 3, "height": 4, "page_width": 5, "page_height": 6, "page_number": 1}
                ],
            },
            {
                "run_name": tenant,
                "extraction_name": extraction_id,
                "tenant": tenant,
                "id": extraction_id,
                "xml_file_name": "one_file_name",
                "values": [{"id": "one_id", "label": "one_label", "segment_text": "one_segment_text"}],
                "segment_text": "one_segment_text",
                "page_number": 2,
                "segments_boxes": [
                    {"left": 1, "top": 2, "width": 3, "height": 4, "page_width": 5, "page_height": 6, "page_number": 2}
                ],
            },
            {
                "run_name": tenant,
                "extraction_name": extraction_id,
                "tenant": tenant,
                "id": extraction_id,
                "xml_file_name": "other_file_name",
                "values": [
                    {"id": "other_id", "label": "other_label", "segment_text": "other_segment_text"},
                    {"id": "other_id_2", "label": "other_label_2", "segment_text": "other_segment_text_2"},
                ],
                "segment_text": "other_segment_text",
                "page_number": 3,
                "segments_boxes": [
                    {"left": 1, "top": 2, "width": 3, "height": 4, "page_width": 5, "page_height": 6, "page_number": 3}
                ],
            },
            {
                "run_name": tenant,
                "extraction_name": "wrong extraction name",
                "tenant": tenant,
                "id": "wrong extraction name",
                "xml_file_name": "other_file_name",
                "values": [{"id": "other_id", "label": "other_label", "segment_text": "other_segment_text"}],
                "segment_text": "other_segment_text",
                "page_number": 4,
                "segments_boxes": [
                    {"left": 1, "top": 2, "width": 3, "height": 4, "page_width": 5, "page_height": 6, "page_number": 4}
                ],
            },
        ]

        mongo_client.pdf_metadata_extraction.suggestions.insert_many(json_data)

        with TestClient(app) as client:
            response = client.get(f"/get_suggestions/{tenant}/{extraction_id}")

        suggestions = json.loads(response.json())

        self.assertEqual(200, response.status_code)
        self.assertEqual(2, len(suggestions))

        self.assertEqual({tenant}, {x["tenant"] for x in suggestions})
        self.assertEqual({extraction_id}, {x["id"] for x in suggestions})

        self.assertEqual("one_file_name", suggestions[0]["xml_file_name"])
        self.assertEqual("one_segment_text", suggestions[0]["segment_text"])
        self.assertEqual(
            [{"id": "one_id", "label": "one_label", "segment_text": "one_segment_text"}], suggestions[0]["values"]
        )
        self.assertEqual(2, suggestions[0]["page_number"])

        self.assertEqual("other_file_name", suggestions[1]["xml_file_name"])
        self.assertEqual("other_segment_text", suggestions[1]["segment_text"])
        self.assertEqual(
            [
                {"id": "other_id", "label": "other_label", "segment_text": "other_segment_text"},
                {"id": "other_id_2", "label": "other_label_2", "segment_text": "other_segment_text_2"},
            ],
            suggestions[1]["values"],
        )
        self.assertEqual(3, suggestions[1]["page_number"])

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_should_remove_suggestions_when_returned(self):
        tenant = "example_tenant_name"
        extraction_id = "prediction_extraction_id"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        json_data = [
            {
                "run_name": tenant + "1",
                "extraction_name": extraction_id,
                "tenant": tenant + "1",
                "id": extraction_id,
                "xml_file_name": "one_file_name",
                "text": "one_text_predicted",
                "segment_text": "one_segment_text",
                "page_number": 1,
                "segments_boxes": [
                    {"left": 1, "top": 2, "width": 3, "height": 4, "page_width": 5, "page_height": 6, "page_number": 1}
                ],
            },
            {
                "run_name": tenant + "2",
                "extraction_name": extraction_id,
                "tenant": tenant + "2",
                "id": extraction_id,
                "xml_file_name": "one_file_name",
                "text": "one_text_predicted",
                "segment_text": "one_segment_text",
                "page_number": 2,
                "segments_boxes": [
                    {"left": 1, "top": 2, "width": 3, "height": 4, "page_width": 5, "page_height": 6, "page_number": 2}
                ],
            },
        ]

        mongo_client.pdf_metadata_extraction.suggestions.insert_many(json_data)

        with TestClient(app) as client:
            client.get(f"/get_suggestions/{tenant}1/{extraction_id}")

        suggestion = Suggestion(**mongo_client.pdf_metadata_extraction.suggestions.find_one())

        suggestions_collection = mongo_client.pdf_metadata_extraction.suggestions
        self.assertEqual(1, suggestions_collection.count_documents({}))
        self.assertEqual(tenant + "2", suggestion.tenant)
        self.assertEqual(extraction_id, suggestion.id)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions_when_no_suggestions(self):
        with TestClient(app) as client:
            response = client.get("/get_suggestions/tenant/property")
        suggestions = json.loads(response.json())

        self.assertEqual(200, response.status_code)
        self.assertEqual(0, len(suggestions))

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_save_suggestions(self):
        tenant = "example_tenant_name"
        extraction_id = "prediction_extraction_id"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        suggestions = [
            Suggestion(
                tenant=tenant,
                id=extraction_id,
                xml_file_name="xml_file_name",
                entity_name="entity_name",
                text="text_predicted",
                segment_text="segment_text",
                page_number=1,
                segments_boxes=[
                    SegmentBox(
                        left=1,
                        top=2,
                        width=3,
                        height=4,
                        page_width=5,
                        page_height=6,
                        page_number=1,
                        segment_type=TokenType.TEXT,
                    )
                ],
            )
        ]

        with TestClient(app) as client:
            response = client.post(f"/save_suggestions/{tenant}/{extraction_id}", json=[s.model_dump() for s in suggestions])

        self.assertEqual(200, response.status_code)

        suggestion_document = mongo_client.pdf_metadata_extraction.suggestions.find_one()

        self.assertEqual(tenant, suggestion_document["tenant"])
        self.assertEqual(extraction_id, suggestion_document["id"])
        self.assertEqual("xml_file_name", suggestion_document["xml_file_name"])
        self.assertEqual("entity_name", suggestion_document["entity_name"])
        self.assertEqual("text_predicted", suggestion_document["text"])
        self.assertEqual("segment_text", suggestion_document["segment_text"])
        self.assertEqual(1, suggestion_document["page_number"])
        self.assertEqual(
            [
                {
                    "left": 1,
                    "top": 2,
                    "width": 3,
                    "height": 4,
                    "page_width": 5,
                    "page_height": 6,
                    "page_number": 1,
                    "segment_type": "Text",
                }
            ],
            suggestion_document["segments_boxes"],
        )

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_samples_training(self):
        tenant = "example_tenant_name"
        extraction_id = "extraction_id"

        labeled_data = [
            {
                "run_name": tenant,
                "extraction_name": extraction_id,
                "tenant": tenant,
                "id": extraction_id,
                "xml_file_name": "",
                "text": "one_text",
                "source_text": "one_text",
                "page_width": 1.1,
                "page_height": 2.1,
                "xml_segments_boxes": [
                    {"left": 1, "top": 2, "width": 3, "height": 4, "page_width": 5, "page_height": 6, "page_number": 5}
                ],
                "label_segments_boxes": [
                    {"left": 8, "top": 12, "width": 16, "height": 20, "page_width": 5, "page_height": 6, "page_number": 10}
                ],
            },
            {
                "run_name": tenant,
                "extraction_name": extraction_id,
                "tenant": tenant,
                "id": extraction_id,
                "xml_file_name": "",
                "text": "other_text",
                "source_text": "other_text",
                "page_width": 3.1,
                "page_height": 4.1,
                "xml_segments_boxes": [],
                "label_segments_boxes": [],
            },
        ]

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")
        mongo_client.pdf_metadata_extraction.labeled_data.insert_many(labeled_data)

        with TestClient(app) as client:
            response = client.get(f"/get_samples_training/{tenant}/{extraction_id}")

        training_samples = [TrainingSample(**x) for x in response.json()]

        self.assertEqual(200, response.status_code)
        self.assertEqual(2, len(training_samples))

        self.assertEqual({tenant}, {x.labeled_data.tenant for x in training_samples})
        self.assertEqual({extraction_id}, {x.labeled_data.id for x in training_samples})

        self.assertEqual("one_text", training_samples[0].labeled_data.source_text)
        self.assertEqual(1.1, training_samples[0].labeled_data.page_width)
        self.assertEqual(2.1, training_samples[0].labeled_data.page_height)
        self.assertEqual(
            [
                SegmentBox(
                    left=1.0,
                    top=2.0,
                    width=3.0,
                    height=4.0,
                    page_number=5,
                    page_width=5,
                    page_height=6,
                    segment_type=TokenType.TEXT,
                )
            ],
            training_samples[0].labeled_data.xml_segments_boxes,
        )
        self.assertEqual(
            [
                SegmentBox(
                    left=8.0,
                    top=12.0,
                    width=16.0,
                    height=20.0,
                    page_number=10,
                    page_width=5,
                    page_height=6,
                    segment_type=TokenType.TEXT,
                )
            ],
            training_samples[0].labeled_data.label_segments_boxes,
        )

        self.assertEqual("other_text", training_samples[1].labeled_data.source_text)
        self.assertEqual(3.1, training_samples[1].labeled_data.page_width)
        self.assertEqual(4.1, training_samples[1].labeled_data.page_height)
        self.assertEqual([], training_samples[1].labeled_data.xml_segments_boxes)
        self.assertEqual([], training_samples[1].labeled_data.label_segments_boxes)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_samples_prediction(self):
        tenant = "example_tenant_name"
        extraction_id = "extraction_id"

        prediction_data = [
            {
                "run_name": tenant,
                "extraction_name": extraction_id,
                "tenant": tenant,
                "id": extraction_id,
                "entity_name": "entity_name",
                "xml_file_name": "",
                "source_text": "one_text",
                "page_width": 1.1,
                "page_height": 2.1,
                "xml_segments_boxes": [],
            },
            {
                "run_name": tenant,
                "extraction_name": extraction_id,
                "tenant": tenant,
                "id": extraction_id,
                "entity_name": "other_entity_name",
                "xml_file_name": "",
                "source_text": "other_text",
                "page_width": 3.1,
                "page_height": 4.1,
                "xml_segments_boxes": [],
            },
        ]

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")
        mongo_client.pdf_metadata_extraction.prediction_data.insert_many(prediction_data)

        with TestClient(app) as client:
            response = client.get(f"/get_samples_prediction/{tenant}/{extraction_id}")

        prediction_samples = [PredictionSample(**x) for x in response.json()]

        self.assertEqual(200, response.status_code)
        self.assertEqual(2, len(prediction_samples))

        self.assertEqual("one_text", prediction_samples[0].source_text)
        self.assertEqual("other_text", prediction_samples[1].source_text)
        self.assertEqual("entity_name", prediction_samples[0].entity_name)
        self.assertEqual("other_entity_name", prediction_samples[1].entity_name)

    def test_delete_extractor(self):
        run_name = "test_run"
        extraction_name = "test_extraction"

        extractor_identifier = ExtractionIdentifier(
            run_name=run_name, extraction_name=extraction_name, output_path=MODELS_DATA_PATH
        )

        self._create_test_extraction_folder(extractor_identifier)

        ExtractorUseCase.remove_old_models(extractor_identifier)

        self.assertTrue(os.path.exists(extractor_identifier.get_path()))
        self.assertLessEqual(4, len(os.listdir(extractor_identifier.get_path())))

        with TestClient(app) as client:
            response = client.delete(f"/{run_name}/{extraction_name}")

        self.assertEqual(200, response.status_code)
        self.assertTrue(response.json())

        ExtractorUseCase.remove_old_models(extractor_identifier)

        self.assertFalse(os.path.exists(extractor_identifier.get_path()))
