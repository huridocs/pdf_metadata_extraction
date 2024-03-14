import json
import time
from unittest import TestCase

from rsmq import RedisSMQ

import requests

from config import APP_PATH
from data.ExtractionTask import ExtractionTask
from data.Option import Option
from data.Params import Params
from data.ResultsMessage import ResultsMessage
from data.Suggestion import Suggestion

ROOT_PATH = "./"

REDIS_HOST = "127.0.0.1"
REDIS_PORT = "6379"

QUEUE = RedisSMQ(
    host=REDIS_HOST,
    port=REDIS_PORT,
    qname="information_extraction_tasks",
    quiet=False,
)

SERVER_URL = "http://127.0.0.1:5056"


class TestEndToEnd(TestCase):
    def test_redis_message_to_ignore(self):
        QUEUE.sendMessage().message('{"message_to_ignore":"to_be_written_in_log_file"}').execute()

    def test_create_model(self):
        tenant = "end_to_end_test"
        extraction_id = "extraction_id"

        test_xml_path = f"{APP_PATH}/tenant_test/extraction_id/xml_to_train/test.xml"
        with open(test_xml_path, mode="rb") as stream:
            files = {"file": stream}
            requests.post(f"{SERVER_URL}/xml_to_train/{tenant}/{extraction_id}", files=files)

        labeled_data_json = {
            "extraction_name": extraction_id,
            "run_name": tenant,
            "xml_file_name": "test.xml",
            "language_iso": "en",
            "label_text": "Original: English",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
            "label_segments_boxes": [{"left": 123, "top": 45, "width": 87, "height": 16, "page_number": 1}],
        }

        requests.post(f"{SERVER_URL}/labeled_data", json=labeled_data_json)

        task = ExtractionTask(
            tenant=tenant,
            task="create_model",
            params=Params(id=extraction_id),
        )
        QUEUE.sendMessage(delay=0).message(task.model_dump_json()).execute()

        results_message = self.get_results_message()
        expected_result = ResultsMessage(
            tenant=tenant,
            task="create_model",
            params=Params(id=extraction_id),
            success=True,
            error_message="",
            data_url=None,
        )

        self.assertEqual(expected_result, results_message)

    def test_get_suggestions(self):
        tenant = "end_to_end_test"
        extraction_id = "extraction_id"

        test_xml_path = f"{APP_PATH}/tenant_test/extraction_id/xml_to_train/test.xml"

        with open(test_xml_path, mode="rb") as stream:
            files = {"file": stream}
            requests.post(f"{SERVER_URL}/xml_to_predict/{tenant}/{extraction_id}", files=files)

        predict_data_json = {
            "run_name": tenant,
            "extraction_name": extraction_id,
            "xml_file_name": "test.xml",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
        }

        requests.post(f"{SERVER_URL}/prediction_data", json=predict_data_json)

        task = ExtractionTask(
            tenant=tenant,
            task="suggestions",
            params=Params(id=extraction_id),
        )
        QUEUE.sendMessage(delay=0).message(str(task.model_dump_json())).execute()

        results_message = self.get_results_message()
        expected_result = ResultsMessage(
            tenant=tenant,
            task="suggestions",
            params=Params(id=extraction_id),
            success=True,
            error_message="",
            data_url=f"{SERVER_URL}/get_suggestions/{tenant}/{extraction_id}",
        )

        self.assertEqual(results_message, expected_result)

        response = requests.get(results_message.data_url)

        suggestions = json.loads(response.json())
        suggestion = Suggestion(**suggestions[0])

        self.assertEqual(1, len(suggestions))

        self.assertEqual(tenant, suggestion.tenant)
        self.assertEqual(extraction_id, suggestion.id)
        self.assertEqual("test.xml", suggestion.xml_file_name)
        self.assertEqual("United Nations", suggestion.text)
        self.assertEqual("United Nations", suggestion.segment_text)
        self.assertEqual(1, suggestion.page_number)

        self.assertEqual(len(suggestion.segments_boxes), 1)
        self.assertEqual(round(123 / 0.75, 0), suggestion.segments_boxes[0].left)
        self.assertEqual(round(45 / 0.75, 0), suggestion.segments_boxes[0].top)
        self.assertEqual(round(87 / 0.75, 0), suggestion.segments_boxes[0].width)
        self.assertEqual(round(16 / 0.75, 0), suggestion.segments_boxes[0].height)
        self.assertEqual(1, suggestion.segments_boxes[0].page_number)

    def test_create_model_error(self):
        tenant = "end_to_end_test"
        extraction_id = "extraction_id"
        task = ExtractionTask(
            tenant=tenant,
            task="create_model",
            params=Params(id=extraction_id),
        )

        QUEUE.sendMessage(delay=0).message(task.model_dump_json()).execute()

        results_message = self.get_results_message()
        expected_result = ResultsMessage(
            tenant=tenant,
            task="create_model",
            params=Params(id=extraction_id),
            success=False,
            error_message="No labeled data to create model",
            data_url=None,
        )

        self.assertEqual(results_message, expected_result)

        task = ExtractionTask(
            tenant=tenant,
            task="suggestions",
            params=Params(id=extraction_id),
        )
        QUEUE.sendMessage(delay=0).message(task.model_dump_json()).execute()

        results_message = self.get_results_message()
        expected_result = ResultsMessage(
            tenant=tenant,
            task="suggestions",
            params=Params(id=extraction_id),
            success=False,
            error_message="No data to calculate suggestions",
            data_url=None,
        )

        self.assertEqual(results_message, expected_result)

    def test_get_suggestions_multi_select(self):
        tenant = "end_to_end_test"
        extraction_id = "multi_select_name"

        test_xml_path = f"{APP_PATH}/tenant_test/extraction_id/xml_to_train/test.xml"

        with open(test_xml_path, mode="rb") as stream:
            files = {"file": stream}
            requests.post(f"{SERVER_URL}/xml_to_train/{tenant}/{extraction_id}", files=files)

        options = [Option(id="1", label="United Nations"), Option(id="2", label="Other")]

        labeled_data_json = {
            "extraction_name": extraction_id,
            "run_name": tenant,
            "xml_file_name": "test.xml",
            "language_iso": "en",
            "options": [{"extraction_name": "1", "label": "United Nations"}],
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
            "label_segments_boxes": [
                {
                    "left": round(123 / 0.75, 0),
                    "top": round(45 / 0.75, 0),
                    "width": round(87 / 0.75, 0),
                    "height": round(16 / 0.75, 0),
                    "page_number": 1,
                }
            ],
        }

        requests.post(f"{SERVER_URL}/labeled_data", json=labeled_data_json)

        with open(test_xml_path, mode="rb") as stream:
            files = {"file": stream}
            requests.post(f"{SERVER_URL}/xml_to_predict/{tenant}/{extraction_id}", files=files)

        predict_data_json = {
            "run_name": tenant,
            "extraction_name": extraction_id,
            "xml_file_name": "test.xml",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
        }

        requests.post(f"{SERVER_URL}/prediction_data", json=predict_data_json)

        task = ExtractionTask(
            tenant=tenant,
            task="create_model",
            params=Params(id=extraction_id, options=options, muti_value=False),
        )

        QUEUE.sendMessage(delay=0).message(task.model_dump_json()).execute()

        self.get_results_message()

        task = ExtractionTask(
            tenant=tenant,
            task="suggestions",
            params=Params(id=extraction_id),
        )

        QUEUE.sendMessage(delay=0).message(task.model_dump_json()).execute()

        results_message = self.get_results_message()
        response = requests.get(results_message.data_url)

        suggestions = json.loads(response.json())
        suggestion = Suggestion(**suggestions[0])

        self.assertEqual(1, len(suggestions))

        self.assertEqual(tenant, suggestion.tenant)
        self.assertEqual(extraction_id, suggestion.id)
        self.assertEqual("test.xml", suggestion.xml_file_name)
        self.assertEqual([Option(id="1", label="United Nations")], suggestion.options)
        self.assertEqual("United Nations", suggestion.segment_text)
        self.assertEqual(1, suggestion.page_number)

        self.assertEqual(len(suggestion.segments_boxes), 1)
        self.assertEqual(round(123 / 0.75, 0), suggestion.segments_boxes[0].left)
        self.assertEqual(round(45 / 0.75, 0), suggestion.segments_boxes[0].top)
        self.assertEqual(round(87 / 0.75, 0), suggestion.segments_boxes[0].width)
        self.assertEqual(round(16 / 0.75, 0), suggestion.segments_boxes[0].height)
        self.assertEqual(1, suggestion.segments_boxes[0].page_number)

    @staticmethod
    def get_results_message() -> ResultsMessage:
        for i in range(20):
            time.sleep(3)
            queue = RedisSMQ(
                host=REDIS_HOST,
                port=REDIS_PORT,
                qname="information_extraction_results",
                quiet=False,
            )
            message = queue.receiveMessage().exceptions(False).execute()
            if message:
                queue.deleteMessage(id=message["extraction_name"]).execute()
                return ResultsMessage(**json.loads(message["message"]))
