import os
from os.path import exists
from time import sleep

import requests
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.PredictionData import PredictionData
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.SegmentationData import SegmentationData
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.use_cases.FilterValidSegmentsPages import FilterValidSegmentsPages
from trainable_entity_extractor.use_cases.XmlFile import XmlFile

from config import SERVICE_HOST, SERVICE_PORT


class SampleProcessorUseCase:
    def __init__(self, extraction_identifier: ExtractionIdentifier):
        self.extraction_identifier = extraction_identifier

    @staticmethod
    def get_samples_for_training(
        extraction_identifier: ExtractionIdentifier, labeled_data_list: list[LabeledData]
    ) -> list[TrainingSample]:
        multi_option_samples: list[TrainingSample] = list()
        page_numbers_list = FilterValidSegmentsPages(extraction_identifier).for_training(labeled_data_list)
        for labeled_data, page_numbers_to_keep in zip(labeled_data_list, page_numbers_list):
            segmentation_data = SegmentationData.from_labeled_data(labeled_data)
            xml_file = XmlFile(
                extraction_identifier=extraction_identifier,
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
            xml_file.delete()

        return multi_option_samples

    @staticmethod
    def get_prediction_samples(
        extractor_identifier: ExtractionIdentifier, prediction_data_list: list[PredictionData] = None
    ) -> list[PredictionSample]:
        filter_valid_pages = FilterValidSegmentsPages(extractor_identifier)
        page_numbers_list = filter_valid_pages.for_prediction(prediction_data_list)
        prediction_samples: list[PredictionSample] = []
        for prediction_data, page_numbers in zip(prediction_data_list, page_numbers_list):
            segmentation_data = SegmentationData.from_prediction_data(prediction_data)
            entity_name = prediction_data.entity_name if prediction_data.entity_name else prediction_data.xml_file_name

            xml_file = XmlFile(
                extraction_identifier=extractor_identifier,
                to_train=False,
                xml_file_name=prediction_data.xml_file_name,
            )

            if exists(xml_file.xml_file_path) and not os.path.isdir(xml_file.xml_file_path):
                pdf_data = PdfData.from_xml_file(xml_file, segmentation_data, page_numbers)
            else:
                pdf_data = PdfData.from_texts([""])

            xml_file.delete()

            sample = PredictionSample(pdf_data=pdf_data, entity_name=entity_name, source_text=prediction_data.source_text)
            prediction_samples.append(sample)

        return prediction_samples

    @staticmethod
    def import_samples(
        extraction_identifier: ExtractionIdentifier, for_training: bool
    ) -> list[TrainingSample | PredictionSample]:
        samples: list[TrainingSample | PredictionSample] = list()
        max_retries = 3
        retry_delay = 5  # seconds

        url = f"{SERVICE_HOST}:{SERVICE_PORT}"
        url += "/get_samples_training" if for_training else "/get_samples_prediction"
        url += f"/{extraction_identifier.run_name}/{extraction_identifier.extraction_name}"

        while True:
            try:
                response = requests.get(url)
                response.raise_for_status()

                if not response.json():
                    break

                samples.extend([TrainingSample(**x) if for_training else PredictionSample(**x) for x in response.json()])
            except requests.exceptions.RequestException as e:
                config_logger.error(f"Error fetching training samples: {e}")
                if max_retries > 0:
                    max_retries -= 1
                    config_logger.info(f"Retrying in {retry_delay} seconds...")
                    sleep(retry_delay)
                    continue
                else:
                    config_logger.error("Max retries reached. Exiting.")
                    break

        return samples

    def get_training_samples(self) -> list[TrainingSample]:
        return self.import_samples(extraction_identifier=self.extraction_identifier, for_training=True)

    def get_prediction_samples_for_suggestions(self) -> list[PredictionSample]:
        return self.import_samples(extraction_identifier=self.extraction_identifier, for_training=False)
