import json
import subprocess
import time
from unittest import TestCase

from rsmq import RedisSMQ

import requests

from data.MetadataExtractionTask import MetadataExtractionTask
from data.Option import Option
from data.Params import Params
from data.ResultsMessage import ResultsMessage
from data.Suggestion import Suggestion

DOCKER_VOLUME_PATH = f"../docker_volume"

REDIS_HOST = "127.0.0.1"
REDIS_PORT = "6379"

QUEUE = RedisSMQ(
    host=REDIS_HOST,
    port=REDIS_PORT,
    qname="information_extraction_tasks",
    quiet=False,
)

SERVER_URL = "http://localhost:5052"


class TestEndToEnd(TestCase):
    @classmethod
    def setUpClass(cls):
        print('hi')
        subprocess.run("../run start:testing -d", shell=True)
        cls.wait_for_the_service()

    @classmethod
    def tearDownClass(cls):
        subprocess.run("../run stop", shell=True)

    def test_redis_message_to_ignore(self):
        QUEUE.sendMessage().message('{"message_to_ignore":"to_be_written_in_log_file"}').execute()

    def test_create_model(self):
        tenant = "end_to_end_test"
        property_name = "property_name"

        test_xml_path = f"{DOCKER_VOLUME_PATH}/tenant_test/property_name/xml_to_train/test.xml"
        with open(test_xml_path, mode="rb") as stream:
            files = {"file": stream}
            requests.post(f"{SERVER_URL}/xml_to_train/{tenant}/{property_name}", files=files)

        labeled_data_json = {
            "property_name": property_name,
            "tenant": tenant,
            "xml_file_name": "test.xml",
            "language_iso": "en",
            "label_text": "text",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
            "label_segments_boxes": [{"left": 165, "top": 64, "width": 111, "height": 17, "page_number": 1}],
        }

        requests.post(f"{SERVER_URL}/labeled_data", json=labeled_data_json)

        task = MetadataExtractionTask(
            tenant=tenant,
            task="create_model",
            params=Params(property_name=property_name),
        )
        QUEUE.sendMessage(delay=0).message(str(task.json())).execute()

        results_message = self.get_results_message()
        expected_result = ResultsMessage(
            tenant=tenant,
            task="create_model",
            params=Params(property_name=property_name),
            success=True,
            error_message="",
            data_url=None,
        )

        self.assertEqual(expected_result, results_message)

    def test_get_suggestions(self):
        tenant = "end_to_end_test"
        property_name = "property_name"

        test_xml_path = f"{DOCKER_VOLUME_PATH}/tenant_test/property_name/xml_to_train/test.xml"

        with open(test_xml_path, mode="rb") as stream:
            files = {"file": stream}
            requests.post(f"{SERVER_URL}/xml_to_predict/{tenant}/{property_name}", files=files)

        predict_data_json = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "test.xml",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
        }

        requests.post(f"{SERVER_URL}/prediction_data", json=predict_data_json)

        task = MetadataExtractionTask(
            tenant=tenant,
            task="suggestions",
            params=Params(property_name=property_name),
        )
        QUEUE.sendMessage(delay=0).message(str(task.json())).execute()

        results_message = self.get_results_message()
        expected_result = ResultsMessage(
            tenant=tenant,
            task="suggestions",
            params=Params(property_name=property_name),
            success=True,
            error_message="",
            data_url=f"{SERVER_URL}/get_suggestions/{tenant}/{property_name}",
        )

        self.assertEqual(results_message, expected_result)

        response = requests.get(results_message.data_url)

        suggestions = json.loads(response.json())
        suggestion = Suggestion(**suggestions[0])

        self.assertEqual(1, len(suggestions))

        self.assertEqual(tenant, suggestion.tenant)
        self.assertEqual(property_name, suggestion.property_name)
        self.assertEqual("test.xml", suggestion.xml_file_name)
        self.assertEqual("United Nations", suggestion.text)
        self.assertEqual("United Nations", suggestion.segment_text)
        self.assertEqual(1, suggestion.page_number)

        self.assertEqual(len(suggestion.segments_boxes), 1)
        self.assertAlmostEqual(123 / 0.75, suggestion.segments_boxes[0].left)
        self.assertAlmostEqual(48 / 0.75, suggestion.segments_boxes[0].top)
        self.assertAlmostEqual(82 / 0.75, suggestion.segments_boxes[0].width)
        self.assertAlmostEqual(12 / 0.75, suggestion.segments_boxes[0].height)
        self.assertAlmostEqual(1, suggestion.segments_boxes[0].page_number)

    def test_get_suggestions_multi_select(self):
        tenant = "end_to_end_test"
        property_name = "multi_select_name"

        test_xml_path = f"{DOCKER_VOLUME_PATH}/tenant_test/property_name/xml_to_train/test.xml"

        with open(test_xml_path, mode="rb") as stream:
            files = {"file": stream}
            requests.post(f"{SERVER_URL}/xml_to_train/{tenant}/{property_name}", files=files)

        options = [Option(id="1", label="United Nations"), Option(id="2", label="Other")]

        labeled_data_json = {
            "property_name": property_name,
            "tenant": tenant,
            "xml_file_name": "test.xml",
            "language_iso": "en",
            "options": [{"id": "1", "label": "United Nations"}],
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
            "label_segments_boxes": [{"left": 165, "top": 64, "width": 111, "height": 17, "page_number": 1}],
        }

        requests.post(f"{SERVER_URL}/labeled_data", json=labeled_data_json)

        with open(test_xml_path, mode="rb") as stream:
            files = {"file": stream}
            requests.post(f"{SERVER_URL}/xml_to_predict/{tenant}/{property_name}", files=files)

        predict_data_json = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "test.xml",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
        }

        requests.post(f"{SERVER_URL}/prediction_data", json=predict_data_json)

        task = MetadataExtractionTask(
            tenant=tenant,
            task="create_model",
            params=Params(property_name=property_name, options=options, muti_value=False),
        )

        QUEUE.sendMessage(delay=0).message(str(task.json())).execute()

        self.get_results_message()

        task = MetadataExtractionTask(
            tenant=tenant,
            task="suggestions",
            params=Params(property_name=property_name),
        )

        QUEUE.sendMessage(delay=0).message(str(task.json())).execute()

        results_message = self.get_results_message()
        response = requests.get(results_message.data_url)

        suggestions = json.loads(response.json())
        suggestion = Suggestion(**suggestions[0])

        self.assertEqual(1, len(suggestions))

        self.assertEqual(tenant, suggestion.tenant)
        self.assertEqual(property_name, suggestion.property_name)
        self.assertEqual("test.xml", suggestion.xml_file_name)
        self.assertEqual([Option(id="1", label="United Nations")], suggestion.options)
        self.assertEqual("United Nations", suggestion.segment_text)
        self.assertEqual(1, suggestion.page_number)

        self.assertEqual(len(suggestion.segments_boxes), 1)
        self.assertAlmostEqual(123 / 0.75, suggestion.segments_boxes[0].left)
        self.assertAlmostEqual(48 / 0.75, suggestion.segments_boxes[0].top)
        self.assertAlmostEqual(82 / 0.75, suggestion.segments_boxes[0].width)
        self.assertAlmostEqual(12 / 0.75, suggestion.segments_boxes[0].height)
        self.assertAlmostEqual(1, suggestion.segments_boxes[0].page_number)

    def test_create_model_error(self):
        tenant = "end_to_end_test"
        property_name = "property_name"
        task = MetadataExtractionTask(
            tenant=tenant,
            task="create_model",
            params=Params(property_name=property_name),
        )

        QUEUE.sendMessage(delay=0).message(str(task.json())).execute()

        results_message = self.get_results_message()
        expected_result = ResultsMessage(
            tenant=tenant,
            task="create_model",
            params=Params(property_name=property_name),
            success=False,
            error_message="No labeled data to create model",
            data_url=None,
        )

        self.assertEqual(results_message, expected_result)

        task = MetadataExtractionTask(
            tenant=tenant,
            task="suggestions",
            params=Params(property_name=property_name),
        )
        QUEUE.sendMessage(delay=0).message(str(task.json())).execute()

        results_message = self.get_results_message()
        expected_result = ResultsMessage(
            tenant=tenant,
            task="suggestions",
            params=Params(property_name=property_name),
            success=False,
            error_message="No data to calculate suggestions",
            data_url=None,
        )

        self.assertEqual(results_message, expected_result)

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
                queue.deleteMessage(id=message["id"]).execute()
                return ResultsMessage(**json.loads(message["message"]))

    @staticmethod
    def wait_for_the_service():
        for i in range(13):
            time.sleep(5)
            if requests.get(f"{SERVER_URL}/info").status_code != 200:
                continue
            try:
                queue = RedisSMQ(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    qname="information_extraction_tasks",
                    quiet=False,
                )

                queue.sendMessage().message('{"try_queue":"true"}').execute()
                break
            except AttributeError:
                print("waiting 5 second for the service to start")

        TestEndToEnd.get_results_message()
