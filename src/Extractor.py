import os
import shutil
from os.path import join, exists
from pathlib import Path
from time import time

import pymongo
from trainable_entity_extractor.FilterValidSegmentsPages import FilterValidSegmentsPages
from trainable_entity_extractor.TrainableEntityExtractor import TrainableEntityExtractor
from trainable_entity_extractor.XmlFile import XmlFile
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.data.ExtractionData import ExtractionData
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.Option import Option
from trainable_entity_extractor.data.PdfData import PdfData
from trainable_entity_extractor.data.PredictionData import PredictionData
from trainable_entity_extractor.data.PredictionSample import PredictionSample
from trainable_entity_extractor.data.SegmentationData import SegmentationData
from trainable_entity_extractor.data.Suggestion import Suggestion
from trainable_entity_extractor.data.TrainingSample import TrainingSample
from trainable_entity_extractor.send_logs import send_logs

from config import MONGO_PORT, MONGO_HOST, DATA_PATH
from data.ExtractionTask import ExtractionTask


class Extractor:
    CREATE_MODEL_TASK_NAME = "create_model"
    SUGGESTIONS_TASK_NAME = "suggestions"

    def __init__(self, extraction_identifier: ExtractionIdentifier, options: list[Option] = None, multi_value: bool = False):
        self.extraction_identifier = extraction_identifier
        self.multi_value = multi_value
        self.options = options
        client = pymongo.MongoClient(f"{MONGO_HOST}:{MONGO_PORT}")
        self.pdf_metadata_extraction_db = client["pdf_metadata_extraction"]
        self.mongo_filter = {"tenant": self.extraction_identifier.run_name, "id": self.extraction_identifier.extraction_name}

    def get_labeled_data(self):
        labeled_data_list = []
        for document in self.pdf_metadata_extraction_db.labeled_data.find(self.mongo_filter):
            labeled_data = LabeledData(**document)
            for segment in labeled_data.xml_segments_boxes + labeled_data.label_segments_boxes:
                segment.page_width = segment.page_width if segment.page_width else labeled_data.page_width
                segment.page_height = segment.page_height if segment.page_height else labeled_data.page_height

            labeled_data_list.append(LabeledData(**document))

        return labeled_data_list

    def get_extraction_data_for_training(self, labeled_data_list: list[LabeledData]) -> ExtractionData:
        multi_option_samples: list[TrainingSample] = list()
        page_numbers_list = FilterValidSegmentsPages(self.extraction_identifier).for_training(labeled_data_list)
        for labeled_data, page_numbers_to_keep in zip(labeled_data_list, page_numbers_list):
            segmentation_data = SegmentationData.from_labeled_data(labeled_data)
            xml_file = XmlFile(
                extraction_identifier=self.extraction_identifier,
                to_train=True,
                xml_file_name=labeled_data.xml_file_name,
            )

            if exists(xml_file.xml_file_path) and not os.path.isdir(xml_file.xml_file_path):
                pdf_data = PdfData.from_xml_file(xml_file, segmentation_data, page_numbers_to_keep)
            else:
                pdf_data = PdfData.from_texts([""])
            sample = TrainingSample(
                pdf_data=pdf_data, labeled_data=labeled_data, segment_selector_texts=[labeled_data.source_text]
            )
            multi_option_samples.append(sample)

        return ExtractionData(
            samples=multi_option_samples,
            options=self.options,
            multi_value=self.multi_value,
            extraction_identifier=self.extraction_identifier,
        )

    def create_models(self) -> (bool, str):
        start = time()
        send_logs(self.extraction_identifier, "Loading data to create model")
        extraction_data: ExtractionData = self.get_extraction_data_for_training(self.get_labeled_data())
        send_logs(self.extraction_identifier, f"Set data in {round(time() - start, 2)} seconds")
        self.delete_training_data()
        trainable_entity_extractor = TrainableEntityExtractor(self.extraction_identifier)
        return trainable_entity_extractor.train(extraction_data)

    def get_prediction_samples(self, prediction_data_list: list[PredictionData] = None) -> list[PredictionSample]:
        filter_valid_pages = FilterValidSegmentsPages(self.extraction_identifier)
        page_numbers_list = filter_valid_pages.for_prediction(prediction_data_list)
        prediction_samples: list[PredictionSample] = []
        for prediction_data, page_numbers in zip(prediction_data_list, page_numbers_list):
            segmentation_data = SegmentationData.from_prediction_data(prediction_data)
            entity_name = prediction_data.entity_name if prediction_data.entity_name else prediction_data.xml_file_name

            xml_file = XmlFile(
                extraction_identifier=self.extraction_identifier,
                to_train=False,
                xml_file_name=prediction_data.xml_file_name,
            )

            if exists(xml_file.xml_file_path) and not os.path.isdir(xml_file.xml_file_path):
                pdf_data = PdfData.from_xml_file(xml_file, segmentation_data, page_numbers)
            else:
                pdf_data = PdfData.from_texts([""])

            sample = PredictionSample(pdf_data=pdf_data, entity_name=entity_name, source_text=prediction_data.source_text)
            prediction_samples.append(sample)

        return prediction_samples

    def get_prediction_data_from_db(self):
        prediction_data_list = []
        for document in self.pdf_metadata_extraction_db.prediction_data.find(self.mongo_filter):
            prediction_data = PredictionData(**document)
            for segment in prediction_data.xml_segments_boxes:
                segment.page_width = segment.page_width if segment.page_width else prediction_data.page_width
                segment.page_height = segment.page_height if segment.page_height else prediction_data.page_height
            prediction_data_list.append(prediction_data)
        return prediction_data_list

    def delete_training_data(self):
        training_xml_path = XmlFile.get_xml_folder_path(extraction_identifier=self.extraction_identifier, to_train=True)
        send_logs(self.extraction_identifier, f"Deleting training data in {training_xml_path}")
        shutil.rmtree(training_xml_path, ignore_errors=True)
        self.pdf_metadata_extraction_db.labeled_data.delete_many(self.mongo_filter)

    def insert_suggestions_in_db(self, suggestions: list[Suggestion]) -> (bool, str):
        if not suggestions:
            return False, "No data to calculate suggestions"

        self.pdf_metadata_extraction_db.suggestions.insert_many([x.to_dict() for x in suggestions])
        xml_folder_path = XmlFile.get_xml_folder_path(extraction_identifier=self.extraction_identifier, to_train=False)
        for suggestion in suggestions:
            entity_name = {"entity_name": suggestion.entity_name, "xml_file_name": ""}
            xml_file_name = {"xml_file_name": suggestion.xml_file_name, "entity_name": ""}
            self.pdf_metadata_extraction_db.prediction_data.delete_many({**self.mongo_filter, **entity_name})
            self.pdf_metadata_extraction_db.prediction_data.delete_many({**self.mongo_filter, **xml_file_name})
            path = Path(join(xml_folder_path, suggestion.xml_file_name))
            if not path.is_dir():
                path.unlink(missing_ok=True)

        return True, ""

    def get_suggestions(self) -> list[Suggestion]:
        prediction_samples = self.get_prediction_samples(self.get_prediction_data_from_db())
        trainable_entity_extractor = TrainableEntityExtractor(self.extraction_identifier)
        return trainable_entity_extractor.predict(prediction_samples)

    @staticmethod
    def remove_old_models(extractor_identifier: ExtractionIdentifier):
        if exists(extractor_identifier.get_path()):
            os.utime(extractor_identifier.get_path())

        for run_name in os.listdir(DATA_PATH):
            if run_name == "cache":
                continue

            for extraction_name in os.listdir(join(DATA_PATH, run_name)):
                extractor_identifier_to_check = ExtractionIdentifier(
                    run_name=run_name, extraction_name=extraction_name, output_path=DATA_PATH
                )
                if extractor_identifier_to_check.is_old():
                    config_logger.info(f"Removing old model folder {extractor_identifier_to_check.get_path()}")
                    shutil.rmtree(extractor_identifier_to_check.get_path(), ignore_errors=True)

    @staticmethod
    def calculate_task(extraction_task: ExtractionTask) -> (bool, str):
        extractor_identifier = ExtractionIdentifier(
            run_name=extraction_task.tenant,
            extraction_name=extraction_task.params.id,
            metadata=extraction_task.params.metadata,
            output_path=DATA_PATH,
        )

        Extractor.remove_old_models(extractor_identifier)

        if extraction_task.task == Extractor.CREATE_MODEL_TASK_NAME:
            if extraction_task.params.options:
                options = extraction_task.params.options
            else:
                options = extractor_identifier.get_options()

            multi_value = extraction_task.params.multi_value
            extractor = Extractor(extractor_identifier, options, multi_value)
            return extractor.create_models()

        if extraction_task.task == Extractor.SUGGESTIONS_TASK_NAME:
            extractor = Extractor(extractor_identifier)
            suggestions = extractor.get_suggestions()
            return extractor.insert_suggestions_in_db(suggestions)

        return False, "Error"
