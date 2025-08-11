import json
import os
import shutil
import time
from unittest import TestCase

from pdf_token_type_labels.TokenType import TokenType
from rsmq import RedisSMQ

import requests
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.SegmentBox import SegmentBox
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.domain.Value import Value

from config import APP_PATH, MODELS_DATA_PATH
from domain.ParagraphExtractionResultsMessage import ParagraphExtractionResultsMessage
from domain.Params import Params
from domain.ResultsMessage import ResultsMessage
from domain.TrainableEntityExtractionTask import TrainableEntityExtractionTask

ROOT_PATH = "../"

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
    def tearDown(self):
        requests.delete(f"{SERVER_URL}/delete_folder/end_to_end_test/extraction_id")
        requests.delete(f"{SERVER_URL}/delete_folder/end_to_end_test/model_without_data")
        requests.delete(f"{SERVER_URL}/delete_folder/end_to_end_test/pdf_to_multi_option")
        requests.delete(f"{SERVER_URL}/delete_folder/end_to_end_test/pdf_to_text")
        requests.delete(f"{SERVER_URL}/delete_folder/end_to_end_test/text_to_multi_option")
        requests.delete(f"{SERVER_URL}/delete_folder/end_to_end_test/text_to_text")

    def test_redis_message_to_ignore(self):
        QUEUE.sendMessage().message('{"message_to_ignore":"to_be_written_in_log_file"}').execute()

    def test_create_model_without_data(self):
        tenant = "end_to_end_test"
        extraction_id = "model_without_data"
        task = TrainableEntityExtractionTask(
            tenant=tenant,
            task="create_model",
            params=Params(id=extraction_id, metadata={"name": "test"}),
        )

        QUEUE.sendMessage(delay=0).message(task.model_dump_json()).execute()

        results_message = self.get_results_message()
        expected_result = ResultsMessage(
            tenant=tenant,
            task="create_model",
            params=Params(id=extraction_id, metadata={"name": "test"}),
            success=False,
            error_message="No data to create model",
            data_url=None,
        )

        self.assertEqual(expected_result, results_message)

        task = TrainableEntityExtractionTask(
            tenant=tenant,
            task="suggestions",
            params=Params(id=extraction_id, metadata={"name": "test"}),
        )
        QUEUE.sendMessage(delay=0).message(task.model_dump_json()).execute()

        results_message = self.get_results_message()
        expected_result = ResultsMessage(
            tenant=tenant,
            task="suggestions",
            params=Params(id=extraction_id, metadata={"name": "test"}),
            success=False,
            error_message="No data to calculate suggestions",
            data_url=None,
        )

        self.assertEqual(expected_result, results_message)

    def test_pdf_to_text(self):
        tenant = "end_to_end_test"
        extraction_id = "pdf_to_text"

        test_xml_path = f"{APP_PATH}/tests/resources/tenant_test/extraction_id/xml_to_train/test.xml"
        with open(test_xml_path, mode="rb") as stream:
            files = {"file": stream}
            requests.post(f"{SERVER_URL}/xml_to_train/{tenant}/{extraction_id}", files=files)

        labeled_data_json = {
            "id": extraction_id,
            "tenant": tenant,
            "xml_file_name": "test.xml",
            "language_iso": "en",
            "label_text": "Original: English",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
            "label_segments_boxes": [{"left": 123, "top": 45, "width": 87, "height": 16, "page_number": 1}],
        }

        requests.post(f"{SERVER_URL}/labeled_data", json=labeled_data_json)

        task = TrainableEntityExtractionTask(
            tenant=tenant,
            task="create_model",
            params=Params(id=extraction_id, metadata={"name": "test"}),
        )
        QUEUE.sendMessage(delay=0).message(task.model_dump_json()).execute()

        results_message = self.get_results_message()
        expected_result = ResultsMessage(
            tenant=tenant,
            task="create_model",
            params=Params(id=extraction_id, metadata={"name": "test"}),
            success=True,
            error_message="",
            data_url=None,
        )

        self.assertEqual(expected_result, results_message)

        with open(test_xml_path, mode="rb") as stream:
            files = {"file": stream}
            requests.post(f"{SERVER_URL}/xml_to_predict/{tenant}/{extraction_id}", files=files)

        predict_data_json = {
            "tenant": tenant,
            "id": extraction_id,
            "xml_file_name": "test.xml",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
        }

        requests.post(f"{SERVER_URL}/prediction_data", json=predict_data_json)

        task = TrainableEntityExtractionTask(
            tenant=tenant,
            task="suggestions",
            params=Params(id=extraction_id, metadata={"name": "test"}),
        )
        QUEUE.sendMessage(delay=0).message(str(task.model_dump_json())).execute()

        results_message = self.get_results_message()
        expected_result = ResultsMessage(
            tenant=tenant,
            task="suggestions",
            params=Params(id=extraction_id, metadata={"name": "test"}),
            success=True,
            error_message="",
            data_url=f"{SERVER_URL}/get_suggestions/{tenant}/{extraction_id}",
        )

        self.assertEqual(expected_result, results_message)

        response = requests.get(results_message.data_url)

        suggestions = json.loads(response.json())
        suggestion = Suggestion(**suggestions[0])

        self.assertEqual(1, len(suggestions))

        self.assertEqual(tenant, suggestion.tenant)
        self.assertEqual(extraction_id, suggestion.id)
        self.assertEqual("test.xml", suggestion.xml_file_name)
        self.assertEqual("Original: English", suggestion.text)
        self.assertTrue("Original: English" in suggestion.segment_text)
        self.assertEqual(1, suggestion.page_number)

        self.assertEqual(len(suggestion.segments_boxes), 2)
        self.assertEqual(529, suggestion.segments_boxes[0].left)
        self.assertEqual(120, suggestion.segments_boxes[0].top)
        self.assertEqual(100, suggestion.segments_boxes[0].width)
        self.assertEqual(15, suggestion.segments_boxes[0].height)
        self.assertEqual(1, suggestion.segments_boxes[0].page_number)

    def test_pdf_to_multi_option(self):
        tenant = "end_to_end_test"
        extraction_id = "pdf_to_multi_option"
        test_xml_path = f"{APP_PATH}/tests/resources/tenant_test/extraction_id/xml_to_train/test.xml"

        for i in range(10):
            new_test_xml_path = f"{MODELS_DATA_PATH}/test_{i}.xml"
            shutil.copyfile(test_xml_path, new_test_xml_path)

            with open(new_test_xml_path, mode="rb") as stream:
                files = {"file": stream}
                requests.post(f"{SERVER_URL}/xml_to_train/{tenant}/{extraction_id}", files=files)

            os.remove(new_test_xml_path)

            labeled_data_json = {
                "id": extraction_id,
                "tenant": tenant,
                "xml_file_name": f"test_{i}.xml",
                "language_iso": "en",
                "values": [{"id": "1", "label": "United Nations"}],
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            }
            requests.post(f"{SERVER_URL}/labeled_data", json=labeled_data_json)

        options = [Option(id="1", label="United Nations"), Option(id="2", label="Other")]
        task = TrainableEntityExtractionTask(
            tenant=tenant,
            task="create_model",
            params=Params(id=extraction_id, multi_value=False, metadata={"name": "test"}, options=options),
        )

        QUEUE.sendMessage(delay=0).message(task.model_dump_json()).execute()

        message = self.get_results_message()

        with open(test_xml_path, mode="rb") as stream:
            files = {"file": stream}
            requests.post(f"{SERVER_URL}/xml_to_predict/{tenant}/{extraction_id}", files=files)

        predict_data_json = {
            "tenant": tenant,
            "id": extraction_id,
            "xml_file_name": "test.xml",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
        }

        requests.post(f"{SERVER_URL}/prediction_data", json=predict_data_json)

        task = TrainableEntityExtractionTask(
            tenant=tenant,
            task="suggestions",
            params=Params(id=extraction_id, metadata={"name": "test"}),
        )

        QUEUE.sendMessage(delay=0).message(task.model_dump_json()).execute()

        results_message = self.get_results_message()
        assert results_message.data_url is not None, "Data URL should not be None"
        response = requests.get(results_message.data_url)

        suggestions = json.loads(response.json())
        suggestion = Suggestion(**suggestions[0])

        self.assertEqual(1, len(suggestions))

        self.assertEqual(tenant, suggestion.tenant)
        self.assertEqual(extraction_id, suggestion.id)
        self.assertEqual("test.xml", suggestion.xml_file_name)
        self.assertEqual("United Nations", suggestion.segment_text)
        self.assertEqual(
            [
                SegmentBox(
                    left=164.0,
                    top=60.0,
                    width=109.0,
                    height=21.0,
                    page_width=0,
                    page_height=0,
                    page_number=1,
                    segment_type=TokenType.TEXT,
                )
            ],
            suggestion.segments_boxes,
        )
        self.assertEqual([Value(id="1", label="United Nations", segment_text="United Nations")], suggestion.values)

    def test_text_to_multi_option(self):
        tenant = "end_to_end_test"
        extraction_id = "text_to_multi_option"

        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        labeled_data_json = {
            "id": extraction_id,
            "tenant": tenant,
            "entity_name": "entity_name_1",
            "language_iso": "en",
            "values": [{"id": "1", "label": "1"}, {"id": "2", "label": "2"}],
            "source_text": "Option 1 Option 2",
        }

        requests.post(f"{SERVER_URL}/labeled_data", json=labeled_data_json)

        labeled_data_json = {
            "id": extraction_id,
            "tenant": tenant,
            "entity_name": "entity_name_2",
            "language_iso": "en",
            "values": [{"id": "2", "label": "2"}],
            "source_text": "Option 2",
        }

        requests.post(f"{SERVER_URL}/labeled_data", json=labeled_data_json)

        task = TrainableEntityExtractionTask(
            tenant=tenant,
            task="create_model",
            params=Params(id=extraction_id, options=options, multi_value=True, metadata={"name": "test"}),
        )

        QUEUE.sendMessage(delay=0).message(task.model_dump_json()).execute()

        predict_data_json = {
            "tenant": tenant,
            "id": extraction_id,
            "source_text": "Option 1",
            "entity_name": "entity_name_1",
        }

        requests.post(f"{SERVER_URL}/prediction_data", json=predict_data_json)

        predict_data_json = {
            "tenant": tenant,
            "id": extraction_id,
            "source_text": "Option 2 Option 3",
            "entity_name": "entity_name_2",
        }

        requests.post(f"{SERVER_URL}/prediction_data", json=predict_data_json)

        self.get_results_message()

        task = TrainableEntityExtractionTask(
            tenant=tenant,
            task="suggestions",
            params=Params(id=extraction_id, metadata={"name": "test"}),
        )

        QUEUE.sendMessage(delay=0).message(task.model_dump_json()).execute()

        results_message = self.get_results_message()
        response = requests.get(results_message.data_url)

        suggestions = json.loads(response.json())
        suggestion_1 = Suggestion(**suggestions[0])
        suggestion_2 = Suggestion(**suggestions[1])

        self.assertEqual(2, len(suggestions))

        self.assertEqual(tenant, suggestion_1.tenant)
        self.assertEqual(extraction_id, suggestion_1.id)
        self.assertEqual("entity_name_1", suggestion_1.entity_name)
        self.assertEqual([Value(id="1", label="1", segment_text="Option 1")], suggestion_1.values)

        self.assertEqual(tenant, suggestion_2.tenant)
        self.assertEqual(extraction_id, suggestion_2.id)
        self.assertEqual("entity_name_2", suggestion_2.entity_name)
        self.assertEqual(
            [
                Value(id="2", label="2", segment_text="Option 2 Option 3"),
                Value(id="3", label="3", segment_text="Option 2 Option 3"),
            ],
            suggestion_2.values,
        )

    def test_text_to_text(self):
        tenant = "end_to_end_test"
        extraction_id = "text_to_text"

        labeled_data_json = {
            "id": extraction_id,
            "tenant": tenant,
            "entity_name": "entity_name_1",
            "language_iso": "en",
            "label_text": "1",
            "source_text": "Option 1",
        }

        requests.post(f"{SERVER_URL}/labeled_data", json=labeled_data_json)

        labeled_data_json = {
            "id": extraction_id,
            "tenant": tenant,
            "entity_name": "entity_name_2",
            "language_iso": "en",
            "label_text": "2",
            "source_text": "Option 2",
        }

        requests.post(f"{SERVER_URL}/labeled_data", json=labeled_data_json)

        task = TrainableEntityExtractionTask(
            tenant=tenant,
            task="create_model",
            params=Params(id=extraction_id, metadata={"name": "test"}),
        )

        QUEUE.sendMessage(delay=0).message(task.model_dump_json()).execute()

        predict_data_json = {
            "tenant": tenant,
            "id": extraction_id,
            "source_text": "Option 1",
            "entity_name": "entity_name_1",
        }

        requests.post(f"{SERVER_URL}/prediction_data", json=predict_data_json)

        predict_data_json = {
            "tenant": tenant,
            "id": extraction_id,
            "source_text": "Option 3",
            "entity_name": "entity_name_2",
        }

        requests.post(f"{SERVER_URL}/prediction_data", json=predict_data_json)

        self.get_results_message()

        task = TrainableEntityExtractionTask(
            tenant=tenant,
            task="suggestions",
            params=Params(id=extraction_id, metadata={"name": "test"}),
        )

        QUEUE.sendMessage(delay=0).message(task.model_dump_json()).execute()

        results_message = self.get_results_message()
        response = requests.get(results_message.data_url)

        suggestions = json.loads(response.json())
        suggestion_1 = Suggestion(**suggestions[0])
        suggestion_2 = Suggestion(**suggestions[1])

        self.assertEqual(2, len(suggestions))

        self.assertEqual(tenant, suggestion_1.tenant)
        self.assertEqual(extraction_id, suggestion_1.id)
        self.assertEqual("Option 1", suggestion_1.segment_text)
        self.assertEqual("entity_name_1", suggestion_1.entity_name)
        self.assertEqual("1", suggestion_1.text)

        self.assertEqual(tenant, suggestion_2.tenant)
        self.assertEqual(extraction_id, suggestion_2.id)
        self.assertEqual("Option 3", suggestion_2.segment_text)
        self.assertEqual("entity_name_2", suggestion_2.entity_name)
        self.assertEqual("3", suggestion_2.text)

    @staticmethod
    def get_results_message() -> ResultsMessage | ParagraphExtractionResultsMessage | None:
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

        return None
