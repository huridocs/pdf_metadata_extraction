import json
import os
import shutil
import time
from pathlib import Path
from unittest import TestCase

from pdf_token_type_labels.TokenType import TokenType
from rsmq import RedisSMQ

import requests
from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.SegmentBox import SegmentBox
from trainable_entity_extractor.data.Suggestion import Suggestion

from config import APP_PATH, DATA_PATH
from domain.ParagraphExtractionData import ParagraphExtractionData, XmlData
from domain.ParagraphExtractionResultsMessage import ParagraphExtractionResultsMessage
from drivers.rest.ParagraphsTranslations import ParagraphsTranslations
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
        requests.delete(f"{SERVER_URL}/end_to_end_test/extraction_id")
        requests.delete(f"{SERVER_URL}/end_to_end_test/pdf_to_multi_option")
        requests.delete(f"{SERVER_URL}/end_to_end_test/text_to_multi_option")

    def test_redis_message_to_ignore(self):
        QUEUE.sendMessage().message('{"message_to_ignore":"to_be_written_in_log_file"}').execute()

    def test_pdf_to_text(self):
        tenant = "end_to_end_test"
        extraction_id = "extraction_id"

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

    def test_create_model_without_data(self):
        tenant = "end_to_end_test"
        extraction_id = "extraction_id"
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

    def test_pdf_to_multi_option(self):
        tenant = "end_to_end_test"
        extraction_id = "pdf_to_multi_option"
        test_xml_path = f"{APP_PATH}/tests/resources/tenant_test/extraction_id/xml_to_train/test.xml"

        for i in range(10):
            new_test_xml_path = f"{DATA_PATH}/test_{i}.xml"
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

        options = [Option(id="1", label="United Nations"), Option(id="2", label="Other")]
        task = TrainableEntityExtractionTask(
            tenant=tenant,
            task="create_model",
            params=Params(id=extraction_id, multi_value=False, metadata={"name": "test"}, options=options),
        )

        QUEUE.sendMessage(delay=0).message(task.model_dump_json()).execute()

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
        self.assertEqual([Option(id="1", label="United Nations")], suggestion.values)

    def test_text_to_multi_option(self):
        tenant = "end_to_end_test"
        extraction_id = "text_to_multi_option"

        options = [Option(id="1", label="1"), Option(id="2", label="2"), Option(id="3", label="3")]

        labeled_data_json = {
            "id": extraction_id,
            "tenant": tenant,
            "entity_name": "entity_name",
            "language_iso": "en",
            "values": [{"id": "1", "label": "1"}, {"id": "2", "label": "2"}],
            "source_text": "Option 1 Option 2",
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
        self.assertEqual([Option(id="1", label="1")], suggestion_1.values)

        self.assertEqual(tenant, suggestion_2.tenant)
        self.assertEqual(extraction_id, suggestion_2.id)
        self.assertEqual("entity_name_2", suggestion_2.entity_name)
        self.assertEqual([Option(id="2", label="2"), Option(id="3", label="3")], suggestion_2.values)

    def test_extract_paragraphs(self):
        response = requests.post(f"{SERVER_URL}/extract_paragraphs", files=[])
        self.assertEqual(422, response.status_code)

        en_xml_path = Path(APP_PATH, "tests", "resources", "test_en.xml")
        fr_xml_path = Path(APP_PATH, "tests", "resources", "test_fr.xml")

        segment_boxes = [
            SegmentBox(left=183, top=72, width=246, height=22, page_number=1, segment_type=TokenType.PAGE_HEADER),
            SegmentBox(left=72, top=151, width=463, height=96, page_number=1, segment_type=TokenType.TEXT),
            SegmentBox(left=72, top=290, width=43, height=12, page_number=1, segment_type=TokenType.TITLE),
            SegmentBox(left=90, top=318, width=76, height=95, page_number=1, segment_type=TokenType.LIST_ITEM),
        ]

        paragraph_extraction_data = ParagraphExtractionData(
            key="key_1",
            xmls=[
                XmlData(xml_file_name="test_en.xml", language="en", is_main_language=True, xml_segments_boxes=segment_boxes),
                XmlData(
                    xml_file_name="test_fr.xml", language="fr", is_main_language=False, xml_segments_boxes=segment_boxes
                ),
            ],
        )

        files = [
            ("json_data", (None, paragraph_extraction_data.model_dump_json())),
            ("xml_files", open(en_xml_path, "rb")),
            ("xml_files", open(fr_xml_path, "rb")),
        ]

        response = requests.post(f"{SERVER_URL}/extract_paragraphs", files=files)
        self.assertEqual(200, response.status_code)
        results_message = self.get_results_message("paragraph_extraction_results")
        self.assertEqual("key_1", results_message.key)
        self.assertEqual(2, len(results_message.xmls))
        self.assertTrue(results_message.success)
        self.assertEqual("", results_message.error_message)
        self.assertEqual(f"{SERVER_URL}/get_paragraphs_translations/key_1", results_message.data_url)

        response = requests.get(results_message.data_url + "force_error")

        self.assertEqual(422, response.status_code)

        response = requests.get(results_message.data_url)

        paragraphs_translations = ParagraphsTranslations(**response.json())
        self.assertEqual("key_1", paragraphs_translations.key)
        self.assertEqual("en", paragraphs_translations.main_language)
        self.assertEqual(["en", "fr"], paragraphs_translations.available_languages)

        self.assertEqual(2, len(paragraphs_translations.paragraphs))

        self.assertEqual(1, paragraphs_translations.paragraphs[0].position)
        self.assertEqual(2, len(paragraphs_translations.paragraphs[0].translations))

        text = """Some text. Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum
has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a
galley of type and scrambled it to make a type specimen book. It has survived not only five
centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was
popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and
more recently with desktop publishing software like Aldus PageMaker including versions of Lorem
Ipsum."""
        self.assertEqual("en", paragraphs_translations.paragraphs[0].translations[0].language)
        self.assertEqual(False, paragraphs_translations.paragraphs[0].translations[0].needs_user_review)
        self.assertEqual(text.split(), paragraphs_translations.paragraphs[0].translations[0].text.split())
        self.assertEqual("fr", paragraphs_translations.paragraphs[0].translations[1].language)
        self.assertEqual(False, paragraphs_translations.paragraphs[0].translations[1].needs_user_review)
        self.assertEqual(text.split(), paragraphs_translations.paragraphs[0].translations[1].text.split())

        text = """● FORMULA ● FOOTNOTE ● LIST ● TABLE ● FIGURE ● TITLE ● TEXT"""
        self.assertEqual("en", paragraphs_translations.paragraphs[1].translations[0].language)
        self.assertEqual(False, paragraphs_translations.paragraphs[1].translations[0].needs_user_review)
        self.assertEqual(text, paragraphs_translations.paragraphs[1].translations[0].text)
        self.assertEqual("fr", paragraphs_translations.paragraphs[1].translations[1].language)
        self.assertEqual(False, paragraphs_translations.paragraphs[1].translations[1].needs_user_review)
        self.assertEqual(text, paragraphs_translations.paragraphs[1].translations[1].text)

    @staticmethod
    def get_results_message(
        queue_name: str = "information_extraction_results",
    ) -> ResultsMessage | ParagraphExtractionResultsMessage | None:
        for i in range(20):
            time.sleep(3)
            queue = RedisSMQ(
                host=REDIS_HOST,
                port=REDIS_PORT,
                qname=queue_name,
                quiet=False,
            )
            message = queue.receiveMessage().exceptions(False).execute()
            if message:
                queue.deleteMessage(id=message["id"]).execute()
                if queue_name == "paragraph_extraction_results":
                    return ParagraphExtractionResultsMessage(**json.loads(message["message"]))
                return ResultsMessage(**json.loads(message["message"]))
