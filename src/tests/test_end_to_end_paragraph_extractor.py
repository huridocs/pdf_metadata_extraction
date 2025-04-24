import json
import time
from pathlib import Path
from unittest import TestCase

from pdf_token_type_labels.TokenType import TokenType

import requests
from rsmq import RedisSMQ
from trainable_entity_extractor.domain.SegmentBox import SegmentBox

from config import APP_PATH
from domain.ParagraphExtractionData import ParagraphExtractionData, XmlData
from domain.ParagraphExtractionResultsMessage import ParagraphExtractionResultsMessage
from drivers.rest.ParagraphsTranslations import ParagraphsTranslations

REDIS_HOST = "127.0.0.1"
REDIS_PORT = "6379"
SERVER_URL = "http://127.0.0.1:5056"


class TestEndToEndParagraphExtractor(TestCase):
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
            queue_name="extract_paragraphs",
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
        results_message = self.get_results_message()
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
    def get_results_message() -> ParagraphExtractionResultsMessage | None:
        for i in range(20):
            time.sleep(3)
            queue = RedisSMQ(
                host=REDIS_HOST,
                port=REDIS_PORT,
                qname="extract_paragraphs_results",
                quiet=False,
            )
            message = queue.receiveMessage().exceptions(False).execute()
            if message:
                queue.deleteMessage(id=message["id"]).execute()
                return ParagraphExtractionResultsMessage(**json.loads(message["message"]))

        return None
