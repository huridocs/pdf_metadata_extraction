import os
import shutil
from os.path import exists, join
from pathlib import Path
from typing import List
from unittest import TestCase

import mongomock
import pymongo

from ServiceConfig import ServiceConfig
from data.MetadataExtractionTask import MetadataExtractionTask
from data.Params import Params
from data.SegmentBox import SegmentBox
from data.Suggestion import Suggestion
from metadata_extraction.MetadataExtraction import MetadataExtraction

DOCKER_VOLUME_PATH = join(Path(__file__).parent, "..", "..", "..", "docker_volume")

service_config = ServiceConfig()


class TestMetadataExtractor(TestCase):
    test_xml_path = f"{DOCKER_VOLUME_PATH}/tenant_test/property_name/xml_to_train/test.xml"
    model_path = f"{DOCKER_VOLUME_PATH}/tenant_test/property_name/segment_predictor_model/model.model"

    @mongomock.patch(servers=[f"mongodb://127.0.0.1:29017"])
    def test_create_model_error_when_blank_document(self):
        tenant = "segment_test"
        property_name = "property_name"

        base_path = join(DOCKER_VOLUME_PATH, tenant, property_name)

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        json_data = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "blank.xml",
            "label_text": "text",
            "language_iso": "en",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
            "label_segments_boxes": [{"left": 123, "top": 48, "width": 83, "height": 12, "page_number": 1}],
        }
        mongo_client.pdf_information_extraction.labeled_data.insert_one(json_data)

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        os.makedirs(f"{base_path}/xml_to_train")
        shutil.copy(self.test_xml_path, f"{base_path}/xml_to_train/test.xml")

        task = MetadataExtractionTask(
            tenant=tenant,
            task=MetadataExtraction.CREATE_MODEL_TASK_NAME,
            params=Params(property_name=property_name),
        )
        task_calculated, error = MetadataExtraction.calculate_task(task)

        self.assertFalse(task_calculated)

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant))

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_create_model_error_when_no_files(self):
        tenant = "error_segment_test"
        property_name = "error_property_name"

        task = MetadataExtractionTask(
            tenant=tenant,
            task=MetadataExtraction.CREATE_MODEL_TASK_NAME,
            params=Params(property_name=property_name),
        )
        task_calculated, error = MetadataExtraction.calculate_task(task)

        self.assertFalse(task_calculated)
        self.assertEqual(error, "No labeled data to create model")

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_create_model_should_do_nothing_when_no_xml(self):
        tenant = "segment_test"
        property_name = "property_name"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")
        json_data = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "not_found.xml_to_train",
            "language_iso": "en",
            "label_text": "text",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
            "label_segments_boxes": [{"left": 125, "top": 247, "width": 319, "height": 29, "page_number": 1}],
        }

        mongo_client.pdf_information_extraction.labeled_data.insert_one(json_data)

        task = MetadataExtractionTask(
            tenant=tenant,
            task=MetadataExtraction.CREATE_MODEL_TASK_NAME,
            params=Params(property_name=property_name),
        )
        MetadataExtraction.calculate_task(task)

        self.assertFalse(os.path.exists(f"{DOCKER_VOLUME_PATH}/segment_test/property_name/xml_to_train"))
        self.assertFalse(
            os.path.exists(f"{DOCKER_VOLUME_PATH}/segment_test/property_name/segment_predictor_model/model.model")
        )

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_calculate_suggestions(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "segment_test"
        property_name = "property_name"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{DOCKER_VOLUME_PATH}/tenant_test", f"{DOCKER_VOLUME_PATH}/{tenant}")
        shutil.rmtree(
            f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/semantic_model",
            ignore_errors=True,
        )

        labeled_data_json = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "test.xml",
            "language_iso": "en",
            "label_text": "text",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
            "label_segments_boxes": [SegmentBox(left=400, top=115, width=74, height=9, page_number=1).dict()],
        }

        mongo_client.pdf_information_extraction.labeled_data.insert_one(labeled_data_json)

        MetadataExtraction.calculate_task(
            MetadataExtractionTask(
                tenant=tenant,
                task=MetadataExtraction.CREATE_MODEL_TASK_NAME,
                params=Params(property_name=property_name),
            )
        )

        to_predict_json = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "test.xml",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
        }

        mongo_client.pdf_information_extraction.predictiondata.insert_one(to_predict_json)

        task = MetadataExtractionTask(
            tenant=tenant,
            task=MetadataExtraction.SUGGESTIONS_TASK_NAME,
            params=Params(property_name=property_name),
        )
        task_calculated, error = MetadataExtraction.calculate_task(task)

        documents_count = mongo_client.pdf_information_extraction.suggestions.count_documents({})
        suggestion = Suggestion(**mongo_client.pdf_information_extraction.suggestions.find_one())

        self.assertTrue(task_calculated)
        self.assertEqual(1, documents_count)

        self.assertEqual(tenant, suggestion.tenant)
        self.assertEqual(property_name, suggestion.property_name)
        self.assertEqual("test.xml", suggestion.xml_file_name)
        self.assertEqual("Original: English", suggestion.segment_text)
        self.assertEqual("Original: English", suggestion.text)
        self.assertEqual(1, suggestion.page_number)

        self.assertEqual(len(suggestion.segments_boxes), 1)
        self.assertAlmostEqual(397 / 0.75, suggestion.segments_boxes[0].left)
        self.assertAlmostEqual(115 / 0.75, suggestion.segments_boxes[0].top)
        self.assertAlmostEqual(74 / 0.75, suggestion.segments_boxes[0].width)
        self.assertAlmostEqual(9 / 0.75, suggestion.segments_boxes[0].height)
        self.assertAlmostEqual(1, suggestion.segments_boxes[0].page_number)

        self.assertIsNone(mongo_client.pdf_information_extraction.predictiondata.find_one())

        self.assertTrue(os.path.exists(f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/xml_to_predict/spanish.xml"))
        self.assertFalse(os.path.exists(f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/xml_to_predict/test.xml"))

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions_page_2(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "segment_test"
        property_name = "property_name"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{DOCKER_VOLUME_PATH}/tenant_test", f"{DOCKER_VOLUME_PATH}/{tenant}")

        labeled_data_json = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "test.xml",
            "language_iso": "en",
            "label_text": "text",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [SegmentBox(left=0, top=130, width=612, height=70, page_number=2).dict()],
            "label_segments_boxes": [SegmentBox(left=300, top=150, width=5, height=5, page_number=2).dict()],
        }

        mongo_client.pdf_information_extraction.labeled_data.insert_one(labeled_data_json)

        MetadataExtraction.calculate_task(
            MetadataExtractionTask(
                tenant=tenant,
                task=MetadataExtraction.CREATE_MODEL_TASK_NAME,
                params=Params(property_name=property_name),
            )
        )

        to_predict_json = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "test.xml",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [SegmentBox(left=0, top=130, width=612, height=70, page_number=2).dict()],
        }

        mongo_client.pdf_information_extraction.predictiondata.insert_one(to_predict_json)

        task_calculated, error = MetadataExtraction.calculate_task(
            MetadataExtractionTask(
                tenant=tenant,
                task=MetadataExtraction.SUGGESTIONS_TASK_NAME,
                params=Params(property_name=property_name),
            )
        )

        documents_count = mongo_client.pdf_information_extraction.suggestions.count_documents({})
        suggestion = Suggestion(**mongo_client.pdf_information_extraction.suggestions.find_one())

        self.assertTrue(task_calculated)
        self.assertEqual(1, documents_count)
        self.assertEqual(tenant, suggestion.tenant)
        self.assertEqual(property_name, suggestion.property_name)
        self.assertEqual("test.xml", suggestion.xml_file_name)
        self.assertTrue("In accordance with paragraph" in suggestion.segment_text)
        self.assertTrue("every four years" in suggestion.text)
        self.assertEqual(2, suggestion.page_number)
        self.assertTrue(os.path.exists(f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/xml_to_predict"))
        self.assertFalse(os.path.exists(f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/xml_to_predict/test.xml"))

        self.assertEqual(len(suggestion.segments_boxes), 1)
        self.assertAlmostEqual(173.33333333333331, suggestion.segments_boxes[0].left)
        self.assertAlmostEqual(168, suggestion.segments_boxes[0].top)
        self.assertAlmostEqual(476, suggestion.segments_boxes[0].width)
        self.assertAlmostEqual(94.666666666, suggestion.segments_boxes[0].height)
        self.assertAlmostEqual(2, suggestion.segments_boxes[0].page_number)

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_semantic_suggestions(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "tenant_to_be_removed"
        property_name = "property_name"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{DOCKER_VOLUME_PATH}/tenant_test", f"{DOCKER_VOLUME_PATH}/{tenant}")

        to_predict_json = [
            {
                "xml_file_name": "test.xml",
                "property_name": property_name,
                "tenant": tenant,
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            },
            {
                "xml_file_name": "test.xml",
                "property_name": property_name,
                "tenant": tenant,
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            },
        ]

        mongo_client.pdf_information_extraction.predictiondata.insert_many(to_predict_json)

        for i in range(7):
            labeled_data_json = {
                "property_name": property_name,
                "tenant": tenant,
                "xml_file_name": "test.xml",
                "language_iso": "en",
                "label_text": "English",
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
                "label_segments_boxes": [SegmentBox(left=397, top=115, width=74, height=9, page_number=1).dict()],
            }

            mongo_client.pdf_information_extraction.labeled_data.insert_one(labeled_data_json)

        MetadataExtraction.calculate_task(
            MetadataExtractionTask(
                tenant=tenant,
                task=MetadataExtraction.CREATE_MODEL_TASK_NAME,
                params=Params(property_name=property_name),
            )
        )

        task_calculated, error = MetadataExtraction.calculate_task(
            MetadataExtractionTask(
                tenant=tenant,
                task=MetadataExtraction.SUGGESTIONS_TASK_NAME,
                params=Params(property_name=property_name),
            )
        )

        suggestions: List[Suggestion] = list()
        find_filter = {"property_name": property_name, "tenant": tenant}
        for document in mongo_client.pdf_information_extraction.suggestions.find(find_filter, no_cursor_timeout=True):
            suggestions.append(Suggestion(**document))

        self.assertTrue(task_calculated)
        self.assertEqual(2, len(suggestions))
        self.assertEqual({tenant}, {x.tenant for x in suggestions})
        self.assertEqual({property_name}, {x.property_name for x in suggestions})
        self.assertEqual({"test.xml"}, {x.xml_file_name for x in suggestions})
        self.assertEqual({"Original: English"}, {x.segment_text for x in suggestions})
        self.assertEqual({"English"}, {x.text for x in suggestions})

        self.assertEqual({1}, set([len(x.segments_boxes) for x in suggestions]))
        self.assertAlmostEqual(397 / 0.75, suggestions[0].segments_boxes[0].left)
        self.assertAlmostEqual(115 / 0.75, suggestions[0].segments_boxes[0].top)
        self.assertAlmostEqual(74 / 0.75, suggestions[0].segments_boxes[0].width)
        self.assertAlmostEqual(9 / 0.75, suggestions[0].segments_boxes[0].height)
        self.assertAlmostEqual(1, suggestions[0].segments_boxes[0].page_number)

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_semantic_suggestions_numeric(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "tenant_to_be_removed"
        property_name = "property_name"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{DOCKER_VOLUME_PATH}/tenant_test", f"{DOCKER_VOLUME_PATH}/{tenant}")

        to_predict_json = [
            {
                "xml_file_name": "test.xml",
                "property_name": property_name,
                "tenant": tenant,
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            }
        ]

        mongo_client.pdf_information_extraction.predictiondata.insert_many(to_predict_json)

        for i in range(7):
            labeled_data_json = {
                "property_name": property_name,
                "tenant": tenant,
                "xml_file_name": "test.xml",
                "language_iso": "en",
                "label_text": "15",
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
                "label_segments_boxes": [SegmentBox(left=397, top=91, width=10, height=9, page_number=1).dict()],
            }

            mongo_client.pdf_information_extraction.labeled_data.insert_one(labeled_data_json)

        MetadataExtraction.calculate_task(
            MetadataExtractionTask(
                tenant=tenant,
                task=MetadataExtraction.CREATE_MODEL_TASK_NAME,
                params=Params(property_name=property_name),
            )
        )

        task_calculated, error = MetadataExtraction.calculate_task(
            MetadataExtractionTask(
                tenant=tenant,
                task=MetadataExtraction.SUGGESTIONS_TASK_NAME,
                params=Params(property_name=property_name),
            )
        )

        suggestions: List[Suggestion] = list()
        find_filter = {"property_name": property_name, "tenant": tenant}
        for document in mongo_client.pdf_information_extraction.suggestions.find(find_filter, no_cursor_timeout=True):
            suggestions.append(Suggestion(**document))

        self.assertTrue(task_calculated)
        self.assertEqual(1, len(suggestions))
        self.assertEqual({tenant}, {x.tenant for x in suggestions})
        self.assertEqual({property_name}, {x.property_name for x in suggestions})
        self.assertEqual({"test.xml"}, {x.xml_file_name for x in suggestions})
        self.assertEqual({"15 February 2021"}, {x.segment_text for x in suggestions})
        self.assertEqual({"15"}, {x.text for x in suggestions})

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_semantic_suggestions_spanish(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "tenant_to_be_removed"
        property_name = "spa"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{DOCKER_VOLUME_PATH}/tenant_test", f"{DOCKER_VOLUME_PATH}/{tenant}")
        shutil.copytree(
            f"{DOCKER_VOLUME_PATH}/{tenant}/property_name/xml_to_train",
            f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/xml_to_train",
        )
        shutil.copytree(
            f"{DOCKER_VOLUME_PATH}/{tenant}/property_name/xml_to_train",
            f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/xml_to_predict",
        )

        samples_number = 20
        for i in range(samples_number):
            labeled_data_json = {
                "tenant": tenant,
                "property_name": property_name,
                "xml_file_name": "spanish.xml",
                "language_iso": "spa",
                "label_text": "día",
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
                "label_segments_boxes": [SegmentBox(left=289, top=206, width=34, height=10, page_number=1).dict()],
            }

            mongo_client.pdf_information_extraction.labeled_data.insert_one(labeled_data_json)
        to_predict_json = [
            {
                "tenant": tenant,
                "property_name": property_name,
                "xml_file_name": "spanish.xml",
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            },
            {
                "property_name": property_name,
                "tenant": tenant,
                "xml_file_name": "spanish.xml",
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            },
        ]

        mongo_client.pdf_information_extraction.predictiondata.insert_many(to_predict_json)

        MetadataExtraction.calculate_task(
            MetadataExtractionTask(
                tenant=tenant,
                task=MetadataExtraction.CREATE_MODEL_TASK_NAME,
                params=Params(property_name=property_name),
            )
        )

        task_calculated, error = MetadataExtraction.calculate_task(
            MetadataExtractionTask(
                tenant=tenant,
                task=MetadataExtraction.SUGGESTIONS_TASK_NAME,
                params=Params(property_name=property_name),
            )
        )

        suggestions: List[Suggestion] = list()
        find_filter = {"property_name": property_name, "tenant": tenant}
        for document in mongo_client.pdf_information_extraction.suggestions.find(find_filter, no_cursor_timeout=True):
            suggestions.append(Suggestion(**document))

        self.assertTrue(task_calculated)
        self.assertEqual(2, len(suggestions))
        self.assertEqual({tenant}, {x.tenant for x in suggestions})
        self.assertEqual({property_name}, {x.property_name for x in suggestions})
        self.assertEqual({"spanish.xml"}, {x.xml_file_name for x in suggestions})
        self.assertEqual({"por día"}, {x.segment_text for x in suggestions})
        self.assertEqual({"día"}, {x.text for x in suggestions})

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions_no_files_error(self):
        tenant = "error_segment_test"
        property_name = "error_property_name"

        base_path = join(DOCKER_VOLUME_PATH, tenant, property_name)

        if not exists(f"{base_path}/segment_predictor_model"):
            os.makedirs(f"{base_path}/segment_predictor_model")

        shutil.copy(self.model_path, f"{base_path}/segment_predictor_model/")

        task = MetadataExtractionTask(
            tenant=tenant,
            task=MetadataExtraction.SUGGESTIONS_TASK_NAME,
            params=Params(property_name=property_name),
        )
        task_calculated, error = MetadataExtraction.calculate_task(task)

        self.assertFalse(task_calculated)
        self.assertEqual(error, "No data to calculate suggestions")

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant))

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions_no_model_error(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "error_segment_test"
        property_name = "error_property_name"

        base_path = join(DOCKER_VOLUME_PATH, tenant, property_name)

        os.makedirs(f"{base_path}/xml_to_predict")
        shutil.copy(self.test_xml_path, f"{base_path}/xml_to_predict/test.xml")

        to_predict_json = [
            {
                "tenant": tenant,
                "property_name": property_name,
                "xml_file_name": "test.xml",
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            }
        ]

        mongo_client.pdf_information_extraction.predictiondata.insert_many(to_predict_json)

        task = MetadataExtractionTask(
            tenant=tenant,
            task=MetadataExtraction.SUGGESTIONS_TASK_NAME,
            params=Params(property_name=property_name),
        )
        task_calculated, error = MetadataExtraction.calculate_task(task)

        self.assertFalse(task_calculated)
        self.assertEqual(error, "No model")

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant))

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions_blank_document(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "segment_test"
        property_name = "property_name"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{DOCKER_VOLUME_PATH}/tenant_test", f"{DOCKER_VOLUME_PATH}/{tenant}")
        shutil.rmtree(
            f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/semantic_model",
            ignore_errors=True,
        )

        to_predict_json = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "blank.xml",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
        }

        mongo_client.pdf_information_extraction.predictiondata.insert_one(to_predict_json)

        task = MetadataExtractionTask(
            tenant=tenant,
            task=MetadataExtraction.CREATE_MODEL_TASK_NAME,
            params=Params(property_name=property_name),
        )
        task_calculated, error = MetadataExtraction.calculate_task(task)

        self.assertFalse(task_calculated)

        self.assertIsNone(mongo_client.pdf_information_extraction.labeled_data.find_one({}))
        self.assertFalse(exists(join(DOCKER_VOLUME_PATH, tenant, property_name, 'xml_to_train')))

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_multi_option_suggestions(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "tenant_to_be_removed"
        property_name = "property_name"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)
        shutil.copytree(f"{DOCKER_VOLUME_PATH}/tenant_test", f"{DOCKER_VOLUME_PATH}/{tenant}")

        to_predict_json = [
            {
                "xml_file_name": "test.xml",
                "property_name": property_name,
                "tenant": tenant,
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
            }
        ]

        mongo_client.pdf_information_extraction.predictiondata.insert_many(to_predict_json)

        options = [{"id": f"id{n}", "label": str(n)} for n in range(16)]
        for i in range(7):
            labeled_data_json = {
                "property_name": property_name,
                "tenant": tenant,
                "xml_file_name": "test.xml",
                "language_iso": "en",
                "options": [{"id": "id15", "label": "15"}],
                "page_width": 612,
                "page_height": 792,
                "xml_segments_boxes": [],
                "label_segments_boxes": [SegmentBox(left=397, top=91, width=10, height=9, page_number=1).dict()],
            }

            mongo_client.pdf_information_extraction.labeled_data.insert_one(labeled_data_json)

        MetadataExtraction.calculate_task(
            MetadataExtractionTask(
                tenant=tenant,
                task=MetadataExtraction.CREATE_MODEL_TASK_NAME,
                params=Params(property_name=property_name, options=options, multi_value=False),
            )
        )

        task_calculated, error = MetadataExtraction.calculate_task(
            MetadataExtractionTask(
                tenant=tenant,
                task=MetadataExtraction.SUGGESTIONS_TASK_NAME,
                params=Params(property_name=property_name),
            )
        )

        suggestions: List[Suggestion] = list()
        find_filter = {"property_name": property_name, "tenant": tenant}
        for document in mongo_client.pdf_information_extraction.suggestions.find(
            find_filter, no_cursor_timeout=True
        ):
            suggestions.append(Suggestion(**document))

        self.assertTrue(task_calculated)
        self.assertEqual(1, len(suggestions))
        self.assertEqual(tenant, suggestions[0].tenant)
        self.assertEqual(property_name, suggestions[0].property_name)
        self.assertEqual("test.xml", suggestions[0].xml_file_name)
        self.assertEqual("15 February 2021", suggestions[0].segment_text)
        self.assertEqual([{"id": "id15", "label": "15"}], suggestions[0].options)

        self.assertIsNone(mongo_client.pdf_information_extraction.labeled_data.find_one({}))
        self.assertFalse(exists(join(DOCKER_VOLUME_PATH, tenant, property_name, 'xml_to_train')))

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)
