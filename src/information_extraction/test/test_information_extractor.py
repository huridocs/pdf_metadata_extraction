import os
import shutil
from typing import List
from unittest import TestCase

import mongomock
import pymongo

from ServiceConfig import ServiceConfig
from data.InformationExtractionTask import InformationExtractionTask
from data.Params import Params
from data.SegmentBox import SegmentBox
from data.Suggestion import Suggestion
from information_extraction.InformationExtraction import InformationExtraction

DOCKER_VOLUME_PATH = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))}/docker_volume"
service_config = ServiceConfig()


class TestInformationExtractor(TestCase):
    test_xml_path = (
        f"{DOCKER_VOLUME_PATH}/tenant_test/property_name/xml_to_train/test.xml"
    )
    model_path = f"{DOCKER_VOLUME_PATH}/tenant_test/property_name/segment_predictor_model/model.model"

    @mongomock.patch(servers=[f"mongodb://127.0.0.1:29017"])
    def test_create_model(self):
        tenant = "segment_test"
        property_name = "property_name"

        base_path = f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        json_data = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "test.xml",
            "label_text": "text",
            "language_iso": "en",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
            "label_segments_boxes": [
                {"left": 123, "top": 48, "width": 83, "height": 12, "page_number": 1}
            ],
        }
        mongo_client.pdf_information_extraction.labeleddata.insert_one(json_data)

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)

        os.makedirs(f"{base_path}/xml_to_train")
        shutil.copy(self.test_xml_path, f"{base_path}/xml_to_train/test.xml")

        task = InformationExtractionTask(
            tenant=tenant,
            task=InformationExtraction.CREATE_MODEL_TASK_NAME,
            params=Params(property_name=property_name),
        )
        task_calculated, error = InformationExtraction.calculate_task(task)

        self.assertTrue(task_calculated)
        self.assertTrue(
            os.path.exists(f"{base_path}/segment_predictor_model/model.model")
        )
        self.assertEqual(
            0, mongo_client.pdf_information_extraction.labeleddata.count_documents({})
        )
        self.assertFalse(os.path.exists(f"{base_path}/xml_to_train/test.xml"))

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}")

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_create_model_different_tenant(self):
        tenant = "different_segment_test"
        property_name = "different_property_name"

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)

        base_path = f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        json_data = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "test.xml",
            "label_text": "text",
            "language_iso": "en",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
            "label_segments_boxes": [
                {"left": 123, "top": 48, "width": 83, "height": 12, "page_number": 1}
            ],
        }
        mongo_client.pdf_information_extraction.labeleddata.insert_one(json_data)

        os.makedirs(f"{base_path}/xml_to_train")
        shutil.copy(self.test_xml_path, f"{base_path}/xml_to_train/test.xml")

        task = InformationExtractionTask(
            tenant=tenant,
            task=InformationExtraction.CREATE_MODEL_TASK_NAME,
            params=Params(property_name=property_name),
        )
        task_calculated, error = InformationExtraction.calculate_task(task)

        self.assertTrue(task_calculated)
        self.assertTrue(
            os.path.exists(f"{base_path}/segment_predictor_model/model.model")
        )
        self.assertEqual(
            0, mongo_client.pdf_information_extraction.labeleddata.count_documents({})
        )
        self.assertFalse(os.path.exists(f"{base_path}/xml_to_train/test.xml"))

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}")

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_create_model_error_when_no_files(self):
        tenant = "error_segment_test"
        property_name = "error_property_name"

        task = InformationExtractionTask(
            tenant=tenant,
            task=InformationExtraction.CREATE_MODEL_TASK_NAME,
            params=Params(property_name=property_name),
        )
        task_calculated, error = InformationExtraction.calculate_task(task)

        self.assertFalse(task_calculated)
        self.assertEqual(error, "No labeled data to create model")

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_create_model_should_remove_previous_models(self):
        tenant = "segment_test"
        property_name = "property_name"

        base_path = f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}"

        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")
        json_data = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "test.xml",
            "label_text": "text",
            "language_iso": "en",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
            "label_segments_boxes": [
                {"left": 123, "top": 48, "width": 83, "height": 12, "page_number": 1}
            ],
        }
        mongo_client.pdf_information_extraction.labeleddata.insert_one(json_data)

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)
        shutil.copytree(
            f"{DOCKER_VOLUME_PATH}/tenant_test", f"{DOCKER_VOLUME_PATH}/{tenant}"
        )

        task = InformationExtractionTask(
            tenant=tenant,
            task=InformationExtraction.CREATE_MODEL_TASK_NAME,
            params=Params(property_name=property_name),
        )

        task_calculated, error = InformationExtraction.calculate_task(task)

        self.assertTrue(task_calculated)
        self.assertTrue(
            os.path.exists(f"{base_path}/segment_predictor_model/model.model")
        )
        self.assertFalse(os.path.exists(f"{base_path}/xml_to_train/test.xml"))
        self.assertFalse(os.path.exists(f"{base_path}/semantic_model/best_model"))

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}")

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
            "label_segments_boxes": [
                {"left": 125, "top": 247, "width": 319, "height": 29, "page_number": 1}
            ],
        }

        mongo_client.pdf_information_extraction.labeleddata.insert_one(json_data)

        task = InformationExtractionTask(
            tenant=tenant,
            task=InformationExtraction.CREATE_MODEL_TASK_NAME,
            params=Params(property_name=property_name),
        )
        InformationExtraction.calculate_task(task)

        self.assertFalse(
            os.path.exists(
                f"{DOCKER_VOLUME_PATH}/segment_test/property_name/xml_to_train"
            )
        )
        self.assertFalse(
            os.path.exists(
                f"{DOCKER_VOLUME_PATH}/segment_test/property_name/segment_predictor_model/model.model"
            )
        )

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_calculate_suggestions(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "segment_test"
        property_name = "property_name"

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)
        shutil.copytree(
            f"{DOCKER_VOLUME_PATH}/tenant_test", f"{DOCKER_VOLUME_PATH}/{tenant}"
        )
        shutil.rmtree(
            f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/semantic_model",
            ignore_errors=True,
        )

        to_predict_json = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "test.xml",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [],
        }

        mongo_client.pdf_information_extraction.predictiondata.insert_one(
            to_predict_json
        )

        task = InformationExtractionTask(
            tenant=tenant,
            task=InformationExtraction.SUGGESTIONS_TASK_NAME,
            params=Params(property_name=property_name),
        )
        task_calculated, error = InformationExtraction.calculate_task(task)

        documents_count = (
            mongo_client.pdf_information_extraction.suggestions.count_documents({})
        )
        suggestion = Suggestion(
            **mongo_client.pdf_information_extraction.suggestions.find_one()
        )

        self.assertTrue(task_calculated)
        self.assertEqual(1, documents_count)

        self.assertEqual(tenant, suggestion.tenant)
        self.assertEqual(property_name, suggestion.property_name)
        self.assertEqual("test.xml", suggestion.xml_file_name)
        self.assertEqual("Original: English", suggestion.segment_text)
        self.assertEqual("Original: English", suggestion.text)
        self.assertEqual(1, suggestion.page_number)

        self.assertIsNone(
            mongo_client.pdf_information_extraction.predictiondata.find_one()
        )

        self.assertTrue(
            os.path.exists(
                f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/xml_to_predict/spanish.xml"
            )
        )
        self.assertFalse(
            os.path.exists(
                f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/xml_to_predict/test.xml"
            )
        )

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions_page_2(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "segment_test"
        property_name = "property_name"

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)
        shutil.copytree(
            f"{DOCKER_VOLUME_PATH}/tenant_test", f"{DOCKER_VOLUME_PATH}/{tenant}"
        )

        labeled_data_json = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "test.xml",
            "language_iso": "en",
            "label_text": "text",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [
                SegmentBox(left=0, top=130, width=612, height=70, page_number=2).dict()
            ],
            "label_segments_boxes": [
                SegmentBox(left=300, top=150, width=5, height=5, page_number=2).dict()
            ],
        }

        mongo_client.pdf_information_extraction.labeleddata.insert_one(
            labeled_data_json
        )

        InformationExtraction.calculate_task(
            InformationExtractionTask(
                tenant=tenant,
                task=InformationExtraction.CREATE_MODEL_TASK_NAME,
                params=Params(property_name=property_name),
            )
        )

        to_predict_json = {
            "tenant": tenant,
            "property_name": property_name,
            "xml_file_name": "test.xml",
            "page_width": 612,
            "page_height": 792,
            "xml_segments_boxes": [
                SegmentBox(left=0, top=130, width=612, height=70, page_number=2).dict()
            ],
        }

        mongo_client.pdf_information_extraction.predictiondata.insert_one(
            to_predict_json
        )

        task_calculated, error = InformationExtraction.calculate_task(
            InformationExtractionTask(
                tenant=tenant,
                task=InformationExtraction.SUGGESTIONS_TASK_NAME,
                params=Params(property_name=property_name),
            )
        )

        documents_count = (
            mongo_client.pdf_information_extraction.suggestions.count_documents({})
        )
        suggestion = Suggestion(
            **mongo_client.pdf_information_extraction.suggestions.find_one()
        )

        self.assertTrue(task_calculated)
        self.assertEqual(1, documents_count)
        self.assertEqual(tenant, suggestion.tenant)
        self.assertEqual(property_name, suggestion.property_name)
        self.assertEqual("test.xml", suggestion.xml_file_name)
        self.assertTrue("In accordance with paragraph" in suggestion.segment_text)
        self.assertTrue("every four years" in suggestion.text)
        self.assertEqual(2, suggestion.page_number)
        self.assertTrue(
            os.path.exists(
                f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/xml_to_predict"
            )
        )
        self.assertFalse(
            os.path.exists(
                f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}/xml_to_predict/test.xml"
            )
        )

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_semantic_suggestions(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "tenant_to_be_removed"
        property_name = "property_name"

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)
        shutil.copytree(
            f"{DOCKER_VOLUME_PATH}/tenant_test", f"{DOCKER_VOLUME_PATH}/{tenant}"
        )

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

        mongo_client.pdf_information_extraction.predictiondata.insert_many(
            to_predict_json
        )

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
                "label_segments_boxes": [
                    SegmentBox(
                        left=397, top=115, width=74, height=9, page_number=1
                    ).dict()
                ],
            }

            mongo_client.pdf_information_extraction.labeleddata.insert_one(
                labeled_data_json
            )

        InformationExtraction.calculate_task(
            InformationExtractionTask(
                tenant=tenant,
                task=InformationExtraction.CREATE_MODEL_TASK_NAME,
                params=Params(property_name=property_name),
            )
        )

        task_calculated, error = InformationExtraction.calculate_task(
            InformationExtractionTask(
                tenant=tenant,
                task=InformationExtraction.SUGGESTIONS_TASK_NAME,
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
        self.assertEqual(2, len(suggestions))
        self.assertEqual({tenant}, {x.tenant for x in suggestions})
        self.assertEqual({property_name}, {x.property_name for x in suggestions})
        self.assertEqual({"test.xml"}, {x.xml_file_name for x in suggestions})
        self.assertEqual({"Original: English"}, {x.segment_text for x in suggestions})
        self.assertEqual({"English"}, {x.text for x in suggestions})

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_semantic_suggestions_numeric(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "tenant_to_be_removed"
        property_name = "property_name"

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)
        shutil.copytree(
            f"{DOCKER_VOLUME_PATH}/tenant_test", f"{DOCKER_VOLUME_PATH}/{tenant}"
        )

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

        mongo_client.pdf_information_extraction.predictiondata.insert_many(
            to_predict_json
        )

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
                "label_segments_boxes": [
                    SegmentBox(
                        left=397, top=91, width=10, height=9, page_number=1
                    ).dict()
                ],
            }

            mongo_client.pdf_information_extraction.labeleddata.insert_one(
                labeled_data_json
            )

        InformationExtraction.calculate_task(
            InformationExtractionTask(
                tenant=tenant,
                task=InformationExtraction.CREATE_MODEL_TASK_NAME,
                params=Params(property_name=property_name),
            )
        )

        task_calculated, error = InformationExtraction.calculate_task(
            InformationExtractionTask(
                tenant=tenant,
                task=InformationExtraction.SUGGESTIONS_TASK_NAME,
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
        self.assertEqual({tenant}, {x.tenant for x in suggestions})
        self.assertEqual({property_name}, {x.property_name for x in suggestions})
        self.assertEqual({"test.xml"}, {x.xml_file_name for x in suggestions})
        self.assertEqual({"15 February 2021"}, {x.segment_text for x in suggestions})
        self.assertEqual({"15"}, {x.text for x in suggestions})

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_semantic_suggestions_spanish(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "tenant_to_be_removed"
        property_name = "spa"

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)
        shutil.copytree(
            f"{DOCKER_VOLUME_PATH}/tenant_test", f"{DOCKER_VOLUME_PATH}/{tenant}"
        )
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
                "label_segments_boxes": [
                    SegmentBox(
                        left=289, top=206, width=34, height=10, page_number=1
                    ).dict()
                ],
            }

            mongo_client.pdf_information_extraction.labeleddata.insert_one(
                labeled_data_json
            )
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

        mongo_client.pdf_information_extraction.predictiondata.insert_many(
            to_predict_json
        )

        InformationExtraction.calculate_task(
            InformationExtractionTask(
                tenant=tenant,
                task=InformationExtraction.CREATE_MODEL_TASK_NAME,
                params=Params(property_name=property_name),
            )
        )

        task_calculated, error = InformationExtraction.calculate_task(
            InformationExtractionTask(
                tenant=tenant,
                task=InformationExtraction.SUGGESTIONS_TASK_NAME,
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
        self.assertEqual(2, len(suggestions))
        self.assertEqual({tenant}, {x.tenant for x in suggestions})
        self.assertEqual({property_name}, {x.property_name for x in suggestions})
        self.assertEqual({"spanish.xml"}, {x.xml_file_name for x in suggestions})
        self.assertEqual({"por día"}, {x.segment_text for x in suggestions})
        self.assertEqual({"día"}, {x.text for x in suggestions})

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions_no_files_error(self):
        tenant = "error_segment_test"
        property_name = "error_property_name"

        base_path = f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}"

        os.makedirs(f"{base_path}/segment_predictor_model")
        shutil.copy(self.model_path, f"{base_path}/segment_predictor_model/")

        task = InformationExtractionTask(
            tenant=tenant,
            task=InformationExtraction.SUGGESTIONS_TASK_NAME,
            params=Params(property_name=property_name),
        )
        task_calculated, error = InformationExtraction.calculate_task(task)

        self.assertFalse(task_calculated)
        self.assertEqual(error, "No data to calculate suggestions")

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}")

    @mongomock.patch(servers=["mongodb://127.0.0.1:29017"])
    def test_get_suggestions_no_model_error(self):
        mongo_client = pymongo.MongoClient("mongodb://127.0.0.1:29017")

        tenant = "error_segment_test"
        property_name = "error_property_name"

        base_path = f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}"

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

        mongo_client.pdf_information_extraction.predictiondata.insert_many(
            to_predict_json
        )

        task = InformationExtractionTask(
            tenant=tenant,
            task=InformationExtraction.SUGGESTIONS_TASK_NAME,
            params=Params(property_name=property_name),
        )
        task_calculated, error = InformationExtraction.calculate_task(task)

        self.assertFalse(task_calculated)
        self.assertEqual(error, "No model")

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}")
