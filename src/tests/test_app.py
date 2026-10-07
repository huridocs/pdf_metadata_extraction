import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from os.path import join
from unittest import TestCase

import psycopg2
from fastapi.testclient import TestClient
from psycopg2.extras import Json

from pdf_token_type_labels.TokenType import TokenType
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.domain.SegmentBox import SegmentBox
from trainable_entity_extractor.domain.TrainingSample import TrainingSample

from adapters.PostgresPersistenceRepository import PostgresPersistenceRepository
from config import APP_PATH, MODELS_DATA_PATH, POSTGRES_DSN
from tests.test_helpers import truncate_all_data

from drivers.rest.app import app


def insert_documents(table: str, documents: list[dict]):
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            for document in documents:
                created_at = document.pop("created_at", None)
                if created_at is not None:
                    cur.execute(
                        f"INSERT INTO {table} (run_name, extraction_name, created_at, data) VALUES (%s, %s, %s, %s)",
                        (document["run_name"], document["extraction_name"], created_at, Json(document)),
                    )
                else:
                    cur.execute(
                        f"INSERT INTO {table} (run_name, extraction_name, data) VALUES (%s, %s, %s)",
                        (document["run_name"], document["extraction_name"], Json(document)),
                    )
    finally:
        conn.close()


def fetch_all_documents(table: str) -> list[dict]:
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT data FROM {table}")
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def count_documents(table: str) -> int:
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {table}")
            return cur.fetchone()[0]
    finally:
        conn.close()


class TestApp(TestCase):
    test_file_path = f"{APP_PATH}/tests/resources/tenant_test/extraction_id/xml_to_predict/test.xml"

    def setUp(self):
        """Set up test environment before each test."""
        # Ensure the Postgres schema exists while no records survive between tests
        PostgresPersistenceRepository()
        truncate_all_data()

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

    def test_post_labeled_data(self):
        tenant = "endpoint_test"
        extraction_id = "extraction_id"

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

        labeled_data_document = fetch_all_documents("labeled_data")[0]

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

    def test_post_labeled_data_different_values(self):
        tenant = "different_endpoint_test"
        extraction_id = "different_extraction_id"

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

        labeled_data_document = fetch_all_documents("labeled_data")[0]

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

    def test_post_labeled_data_multi_option(self):
        tenant = "endpoint_test"
        extraction_id = "extraction_id"

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

        labeled_data_document = fetch_all_documents("labeled_data")[0]

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

    def test_post_prediction_data(self):
        tenant = "endpoint_test"
        extraction_id = "extraction_id"

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

        prediction_data_document = fetch_all_documents("prediction_data")[0]

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

    def test_get_suggestions(self):
        tenant = "example_tenant_name"
        extraction_id = "prediction_extraction_id"

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

        insert_documents("suggestions", json_data)

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

    def test_get_suggestions_multi_option(self):
        tenant = "example_tenant_name"
        extraction_id = "prediction_extraction_id"

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

        insert_documents("suggestions", json_data)

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

    def test_suggestions_kept_when_returned(self):
        tenant = "example_tenant_name"
        extraction_id = "prediction_extraction_id"

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

        insert_documents("suggestions", json_data)

        with TestClient(app) as client:
            response = client.get(f"/get_suggestions/{tenant}1/{extraction_id}")

        suggestions = json.loads(response.json())

        self.assertEqual(200, response.status_code)
        self.assertEqual(1, len(suggestions))
        self.assertEqual(2, count_documents("suggestions"))

        with TestClient(app) as client:
            second_response = client.get(f"/get_suggestions/{tenant}1/{extraction_id}")

        second_suggestions = json.loads(second_response.json())
        self.assertEqual(200, second_response.status_code)
        self.assertEqual(suggestions, second_suggestions)

    def test_expired_suggestions_deleted_when_queried(self):
        tenant = "example_tenant_name"
        extraction_id = "prediction_extraction_id"

        now = datetime.now(timezone.utc)

        json_data = [
            {
                "run_name": tenant,
                "extraction_name": extraction_id,
                "tenant": tenant,
                "id": extraction_id,
                "xml_file_name": "expired_file_name",
                "text": "expired_text_predicted",
                "segment_text": "expired_segment_text",
                "page_number": 1,
                "created_at": now - timedelta(hours=7),
            },
            {
                "run_name": tenant,
                "extraction_name": extraction_id,
                "tenant": tenant,
                "id": extraction_id,
                "xml_file_name": "fresh_file_name",
                "text": "fresh_text_predicted",
                "segment_text": "fresh_segment_text",
                "page_number": 2,
                "created_at": now - timedelta(minutes=5),
            },
            {
                "run_name": tenant,
                "extraction_name": extraction_id,
                "tenant": tenant,
                "id": extraction_id,
                "xml_file_name": "legacy_file_name",
                "text": "legacy_text_predicted",
                "segment_text": "legacy_segment_text",
                "page_number": 3,
            },
        ]

        insert_documents("suggestions", json_data)

        with TestClient(app) as client:
            response = client.get(f"/get_suggestions/{tenant}/{extraction_id}")

        suggestions = json.loads(response.json())

        self.assertEqual(200, response.status_code)
        self.assertEqual(2, len(suggestions))
        self.assertEqual({"fresh_file_name", "legacy_file_name"}, {x["xml_file_name"] for x in suggestions})
        self.assertEqual(2, count_documents("suggestions"))

    def test_save_suggestions_deletes_expired_ones(self):
        tenant = "example_tenant_name"
        extraction_id = "prediction_extraction_id"

        now = datetime.now(timezone.utc)

        expired_suggestion = {
            "run_name": tenant,
            "extraction_name": extraction_id,
            "tenant": tenant,
            "id": extraction_id,
            "xml_file_name": "expired_file_name",
            "text": "expired_text_predicted",
            "segment_text": "expired_segment_text",
            "page_number": 1,
            "created_at": now - timedelta(hours=7),
        }

        insert_documents("suggestions", [expired_suggestion])

        suggestion = Suggestion(
            tenant=tenant,
            id=extraction_id,
            xml_file_name="xml_file_name",
            entity_name="entity_name",
            text="text_predicted",
            segment_text="segment_text",
            page_number=1,
        )

        with TestClient(app) as client:
            response = client.post(f"/save_suggestions/{tenant}/{extraction_id}", json=[suggestion.model_dump()])

        self.assertEqual(200, response.status_code)

        remaining = fetch_all_documents("suggestions")

        self.assertEqual(1, len(remaining))
        self.assertEqual("xml_file_name", remaining[0]["xml_file_name"])
        self.assertIsNotNone(remaining[0].get("created_at"))

    def test_get_suggestions_when_no_suggestions(self):
        with TestClient(app) as client:
            response = client.get("/get_suggestions/tenant/property")
        suggestions = json.loads(response.json())

        self.assertEqual(200, response.status_code)
        self.assertEqual(0, len(suggestions))

    def test_save_suggestions(self):
        tenant = "example_tenant_name"
        extraction_id = "prediction_extraction_id"

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

        suggestion_document = fetch_all_documents("suggestions")[0]

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

    def test_save_suggestions_replaces_previous_for_same_key(self):
        tenant = "example_tenant_name"
        extraction_id = "prediction_extraction_id"

        previous_suggestions = [
            {
                "run_name": tenant,
                "extraction_name": extraction_id,
                "tenant": tenant,
                "id": extraction_id,
                "xml_file_name": "previous_file_name",
                "text": "previous_text_predicted",
                "segment_text": "previous_segment_text",
                "page_number": 1,
            },
            {
                "run_name": tenant,
                "extraction_name": "other_extraction_id",
                "tenant": tenant,
                "id": "other_extraction_id",
                "xml_file_name": "other_extraction_file_name",
                "text": "other_extraction_text_predicted",
                "segment_text": "other_extraction_segment_text",
                "page_number": 1,
            },
        ]

        insert_documents("suggestions", previous_suggestions)

        new_suggestions = [
            Suggestion(
                tenant=tenant,
                id=extraction_id,
                xml_file_name="new_file_name",
                entity_name="entity_name",
                text="new_text_predicted",
                segment_text="new_segment_text",
                page_number=2,
            )
        ]

        with TestClient(app) as client:
            response = client.post(
                f"/save_suggestions/{tenant}/{extraction_id}", json=[s.model_dump() for s in new_suggestions]
            )
            second_response = client.get(f"/get_suggestions/{tenant}/{extraction_id}")

        self.assertEqual(200, response.status_code)

        suggestions = json.loads(second_response.json())

        self.assertEqual(1, len(suggestions))
        self.assertEqual("new_file_name", suggestions[0]["xml_file_name"])
        self.assertEqual(2, count_documents("suggestions"))

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

        insert_documents("labeled_data", labeled_data)

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

    def test_get_samples_training_with_cache(self):
        tenant = "cache_test_tenant"
        extraction_id = "cache_test_extraction"

        labeled_data_samples = [
            {
                "run_name": tenant,
                "extraction_name": extraction_id,
                "tenant": tenant,
                "id": extraction_id,
                "xml_file_name": "test_file.xml",
                "label_text": "first_sample_text",
                "page_width": 612.0,
                "page_height": 792.0,
                "xml_segments_boxes": [
                    {
                        "left": 10,
                        "top": 20,
                        "width": 30,
                        "height": 40,
                        "page_width": 612,
                        "page_height": 792,
                        "page_number": 1,
                    }
                ],
                "label_segments_boxes": [
                    {
                        "left": 50,
                        "top": 60,
                        "width": 70,
                        "height": 80,
                        "page_width": 612,
                        "page_height": 792,
                        "page_number": 1,
                    }
                ],
            },
            {
                "run_name": tenant,
                "extraction_name": extraction_id,
                "tenant": tenant,
                "id": extraction_id,
                "xml_file_name": "test_file2.xml",
                "label_text": "second_sample_text",
                "page_width": 612.0,
                "page_height": 792.0,
                "xml_segments_boxes": [
                    {
                        "left": 15,
                        "top": 25,
                        "width": 35,
                        "height": 45,
                        "page_width": 612,
                        "page_height": 792,
                        "page_number": 2,
                    }
                ],
                "label_segments_boxes": [
                    {
                        "left": 55,
                        "top": 65,
                        "width": 75,
                        "height": 85,
                        "page_width": 612,
                        "page_height": 792,
                        "page_number": 2,
                    }
                ],
            },
        ]

        with TestClient(app) as client:
            for sample_data in labeled_data_samples:
                response = client.post("/labeled_data", json=sample_data)
                self.assertEqual(200, response.status_code)

            first_response = client.get(f"/get_samples_training/{tenant}/{extraction_id}")
            self.assertEqual(200, first_response.status_code)
            first_samples = first_response.json()

            second_response = client.get(f"/get_samples_training/{tenant}/{extraction_id}")
            self.assertEqual(200, second_response.status_code)
            second_samples = second_response.json()

        self.assertEqual(2, len(first_samples))
        self.assertEqual(2, len(second_samples))

        self.assertEqual(first_samples, second_samples)

        first_training_samples = [TrainingSample(**x) for x in first_samples]
        second_training_samples = [TrainingSample(**x) for x in second_samples]

        self.assertEqual({tenant}, {x.labeled_data.tenant for x in first_training_samples})
        self.assertEqual({extraction_id}, {x.labeled_data.id for x in first_training_samples})

        self.assertEqual("first_sample_text", first_training_samples[0].labeled_data.label_text)
        self.assertEqual("second_sample_text", first_training_samples[1].labeled_data.label_text)

        self.assertEqual(
            first_training_samples[0].labeled_data.label_text, second_training_samples[0].labeled_data.label_text
        )
        self.assertEqual(
            first_training_samples[1].labeled_data.label_text, second_training_samples[1].labeled_data.label_text
        )

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

        insert_documents("prediction_data", prediction_data)

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

        self.assertTrue(os.path.exists(extractor_identifier.get_path()))
        self.assertLessEqual(4, len(os.listdir(extractor_identifier.get_path())))

        with TestClient(app) as client:
            response = client.delete(f"/{run_name}/{extraction_name}")

        self.assertEqual(200, response.status_code)
        self.assertTrue(response.json())

        self.assertFalse(os.path.exists(extractor_identifier.get_path()))
