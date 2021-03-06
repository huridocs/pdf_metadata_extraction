import json
import os
import shutil
from os.path import join

import mongomock
import pymongo
from fastapi.testclient import TestClient
from unittest import TestCase
from app import app
from data.Suggestion import Suggestion

client = TestClient(app)

DOCKER_VOLUME_PATH = "../docker_volume"


class TestApp(TestCase):
    test_file_path = f"{DOCKER_VOLUME_PATH}/tenant_test/property_name/xml_to_predict/test.xml"

    def test_info(self):
        response = client.get("/info")
        self.assertEqual(200, response.status_code)

    def test_post_train_xml_file(self):
        tenant = "endpoint_test"
        property_name = "property_name"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        with open(self.test_file_path, "rb") as stream:
            files = {"file": stream}
            response = client.post(f"/xml_to_train/{tenant}/{property_name}", files=files)

        self.assertEqual(200, response.status_code)
        to_train_xml_path = f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/xml_to_train/test.xml"
        self.assertTrue(os.path.exists(to_train_xml_path))

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

    def test_post_xml_to_predict(self):
        tenant = "endpoint_test"
        property_name = "property_name"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        with open(self.test_file_path, "rb") as stream:
            files = {"file": stream}
            response = client.post(f"/xml_to_predict/{tenant}/{property_name}", files=files)

        self.assertEqual(200, response.status_code)
        to_train_xml_path = f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/xml_to_predict/test.xml"
        self.assertTrue(os.path.exists(to_train_xml_path))

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_post_labeled_data(self):
        tenant = "endpoint_test"
        property_name = "property_name"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        json_data = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "xml_file_name",
            "language_iso": "en",
            "label_text": "text",
            "page_width": 1.1,
            "page_height": 2.1,
            "xml_segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 5}],
            "label_segments_boxes": [{"left": 8, "top": 12, "width": 16, "height": 20, "page_number": 10}],
        }

        response = client.post("/labeled_data", json=json_data)

        labeled_data_document = mongo_client.pdf_information_extraction.labeled_data.find_one()

        self.assertEqual(200, response.status_code)
        self.assertEqual(tenant, labeled_data_document["tenant"])
        self.assertEqual(property_name, labeled_data_document["property_name"])
        self.assertEqual("text", labeled_data_document["label_text"])
        self.assertEqual("en", labeled_data_document["language_iso"])
        self.assertEqual(1.1, labeled_data_document["page_width"])
        self.assertEqual(2.1, labeled_data_document["page_height"])
        self.assertEqual("xml_file_name", labeled_data_document["xml_file_name"])
        self.assertEqual(
            [{"height": 4.0, "left": 1.0, "page_number": 5, "top": 2.0, "width": 3.0}],
            labeled_data_document["xml_segments_boxes"],
        )
        self.assertEqual(
            [{"height": 15.0, "left": 6.0, "page_number": 10, "top": 9.0, "width": 12.0}],
            labeled_data_document["label_segments_boxes"],
        )

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_post_labeled_data_different_values(self):
        tenant = "different_endpoint_test"
        property_name = "different_property_name"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        json_data = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "different_xml_file_name",
            "language_iso": "spa",
            "label_text": "other_text",
            "page_width": 3.1,
            "page_height": 4.1,
            "xml_segments_boxes": [],
            "label_segments_boxes": [],
        }

        response = client.post("/labeled_data", json=json_data)

        labeled_data_document = mongo_client.pdf_information_extraction.labeled_data.find_one()

        self.assertEqual(200, response.status_code)
        self.assertEqual(tenant, labeled_data_document["tenant"])
        self.assertEqual(property_name, labeled_data_document["property_name"])
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
        property_name = "property_name"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        options_json = [{"id": "id1", "label": "label1"}, {"id": "id2", "label": "label2"}]

        json_data = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "xml_file_name",
            "language_iso": "en",
            "options": options_json,
            "page_width": 1.1,
            "page_height": 2.1,
            "xml_segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 5}],
            "label_segments_boxes": [{"left": 8, "top": 12, "width": 16, "height": 20, "page_number": 10}],
        }

        response = client.post("/labeled_data", json=json_data)

        labeled_data_document = mongo_client.pdf_information_extraction.labeled_data.find_one()

        self.assertEqual(200, response.status_code)
        self.assertEqual(tenant, labeled_data_document["tenant"])
        self.assertEqual(property_name, labeled_data_document["property_name"])
        self.assertEqual(options_json, labeled_data_document["options"])
        self.assertEqual("en", labeled_data_document["language_iso"])
        self.assertEqual(1.1, labeled_data_document["page_width"])
        self.assertEqual(2.1, labeled_data_document["page_height"])
        self.assertEqual("xml_file_name", labeled_data_document["xml_file_name"])
        self.assertEqual(
            [{"height": 4.0, "left": 1.0, "page_number": 5, "top": 2.0, "width": 3.0}],
            labeled_data_document["xml_segments_boxes"],
        )
        self.assertEqual(
            [{"height": 15.0, "left": 6.0, "page_number": 10, "top": 9.0, "width": 12.0}],
            labeled_data_document["label_segments_boxes"],
        )

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_post_prediction_data(self):
        tenant = "endpoint_test"
        property_name = "property_name"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        json_data = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "xml_file_name",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [{"left": 6, "top": 7, "width": 8, "height": 9, "page_number": 10}],
        }

        response = client.post("/prediction_data", json=json_data)

        prediction_data_document = mongo_client.pdf_information_extraction.predictiondata.find_one()

        self.assertEqual(200, response.status_code)
        self.assertEqual(tenant, prediction_data_document["tenant"])
        self.assertEqual(property_name, prediction_data_document["property_name"])
        self.assertEqual(612, prediction_data_document["page_width"])
        self.assertEqual(792, prediction_data_document["page_height"])
        self.assertEqual("xml_file_name", prediction_data_document["xml_file_name"])
        self.assertEqual(
            [{"left": 6, "top": 7, "width": 8, "height": 9, "page_number": 10}],
            prediction_data_document["xml_segments_boxes"],
        )

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions(self):
        tenant = "example_tenant_name"
        property_name = "prediction_property_name"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        json_data = [
            {
                "tenant": "wrong tenant",
                "property_name": property_name,
                "xml_file_name": "one_file_name",
                "text": "one_text_predicted",
                "segment_text": "one_segment_text",
                "page_number": 1,
                "segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 1}],
            },
            {
                "tenant": tenant,
                "property_name": property_name,
                "xml_file_name": "one_file_name",
                "text": "one_text_predicted",
                "segment_text": "one_segment_text",
                "page_number": 2,
                "segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 2}],
            },
            {
                "tenant": tenant,
                "property_name": property_name,
                "xml_file_name": "other_file_name",
                "text": "other_text_predicted",
                "segment_text": "other_segment_text",
                "page_number": 3,
                "segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 3}],
            },
            {
                "tenant": tenant,
                "property_name": "wrong extraction name",
                "xml_file_name": "other_file_name",
                "text": "other_text_predicted",
                "segment_text": "other_segment_text",
                "page_number": 4,
                "segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 4}],
            },
        ]

        mongo_client.pdf_information_extraction.suggestions.insert_many(json_data)

        response = client.get(f"/get_suggestions/{tenant}/{property_name}")
        suggestions = json.loads(response.json())

        self.assertEqual(200, response.status_code)
        self.assertEqual(2, len(suggestions))

        self.assertEqual({tenant}, {x["tenant"] for x in suggestions})
        self.assertEqual({property_name}, {x["property_name"] for x in suggestions})

        self.assertEqual("one_file_name", suggestions[0]["xml_file_name"])
        self.assertEqual("one_segment_text", suggestions[0]["segment_text"])
        self.assertEqual("one_text_predicted", suggestions[0]["text"])
        self.assertEqual(2, suggestions[0]["page_number"])

        self.assertEqual("other_file_name", suggestions[1]["xml_file_name"])
        self.assertEqual("other_segment_text", suggestions[1]["segment_text"])
        self.assertEqual("other_text_predicted", suggestions[1]["text"])
        self.assertEqual(3, suggestions[1]["page_number"])

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions_multi_option(self):
        tenant = "example_tenant_name"
        property_name = "prediction_property_name"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        json_data = [
            {
                "tenant": "wrong tenant",
                "property_name": property_name,
                "xml_file_name": "one_file_name",
                "options": [{"id": "one_id", "label": "one_label"}],
                "segment_text": "one_segment_text",
                "page_number": 1,
                "segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 1}],
            },
            {
                "tenant": tenant,
                "property_name": property_name,
                "xml_file_name": "one_file_name",
                "options": [{"id": "one_id", "label": "one_label"}],
                "segment_text": "one_segment_text",
                "page_number": 2,
                "segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 2}],
            },
            {
                "tenant": tenant,
                "property_name": property_name,
                "xml_file_name": "other_file_name",
                "options": [{"id": "other_id", "label": "other_label"}, {"id": "other_id_2", "label": "other_label_2"}],
                "segment_text": "other_segment_text",
                "page_number": 3,
                "segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 3}],
            },
            {
                "tenant": tenant,
                "property_name": "wrong extraction name",
                "xml_file_name": "other_file_name",
                "options": [{"id": "other_id", "label": "other_label"}],
                "segment_text": "other_segment_text",
                "page_number": 4,
                "segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 4}],
            },
        ]

        mongo_client.pdf_information_extraction.suggestions.insert_many(json_data)

        response = client.get(f"/get_suggestions/{tenant}/{property_name}")
        suggestions = json.loads(response.json())

        self.assertEqual(200, response.status_code)
        self.assertEqual(2, len(suggestions))

        self.assertEqual({tenant}, {x["tenant"] for x in suggestions})
        self.assertEqual({property_name}, {x["property_name"] for x in suggestions})

        self.assertEqual("one_file_name", suggestions[0]["xml_file_name"])
        self.assertEqual("one_segment_text", suggestions[0]["segment_text"])
        self.assertEqual([{"id": "one_id", "label": "one_label"}], suggestions[0]["options"])
        self.assertEqual(2, suggestions[0]["page_number"])

        self.assertEqual("other_file_name", suggestions[1]["xml_file_name"])
        self.assertEqual("other_segment_text", suggestions[1]["segment_text"])
        self.assertEqual(
            [{"id": "other_id", "label": "other_label"}, {"id": "other_id_2", "label": "other_label_2"}],
            suggestions[1]["options"],
        )
        self.assertEqual(3, suggestions[1]["page_number"])

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_should_remove_suggestions_when_returned(self):
        tenant = "example_tenant_name"
        property_name = "prediction_property_name"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        json_data = [
            {
                "tenant": tenant + "1",
                "property_name": property_name,
                "xml_file_name": "one_file_name",
                "text": "one_text_predicted",
                "segment_text": "one_segment_text",
                "page_number": 1,
                "segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 1}],
            },
            {
                "tenant": tenant + "2",
                "property_name": property_name,
                "xml_file_name": "one_file_name",
                "text": "one_text_predicted",
                "segment_text": "one_segment_text",
                "page_number": 2,
                "segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 2}],
            },
        ]

        mongo_client.pdf_information_extraction.suggestions.insert_many(json_data)

        client.get(f"/get_suggestions/{tenant}1/{property_name}")

        suggestion = Suggestion(**mongo_client.pdf_information_extraction.suggestions.find_one())

        suggestions_collection = mongo_client.pdf_information_extraction.suggestions
        self.assertEqual(1, suggestions_collection.count_documents({}))
        self.assertEqual(tenant + "2", suggestion.tenant)
        self.assertEqual(property_name, suggestion.property_name)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions_when_no_suggestions(self):
        response = client.get("/get_suggestions/tenant/property")
        suggestions = json.loads(response.json())

        self.assertEqual(200, response.status_code)
        self.assertEqual(0, len(suggestions))
