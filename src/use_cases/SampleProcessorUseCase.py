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
from trainable_entity_extractor.domain.XmlFile import XmlFile
from trainable_entity_extractor.use_cases.FilterValidSegmentsPagesUseCase import FilterValidSegmentsPagesUseCase

from config import SERVICE_HOST, SERVICE_PORT
from use_cases.SamplesCacheUseCase import SamplesCacheUseCase


class SampleProcessorUseCase:
    def __init__(self, extractor_identifier: ExtractionIdentifier):
        self.extraction_identifier = extractor_identifier
        self.samples_cache_use_case = SamplesCacheUseCase()

    def get_samples_for_training(self, labeled_data_list: list[LabeledData]) -> list[TrainingSample]:
        multi_option_samples: list[TrainingSample] = list()
        page_numbers_list = FilterValidSegmentsPagesUseCase(self.extraction_identifier).for_training(labeled_data_list)
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
            xml_file.delete()

        return multi_option_samples

    def get_prediction_samples(self, prediction_data_list: list[PredictionData] = None) -> list[PredictionSample]:
        filter_valid_pages = FilterValidSegmentsPagesUseCase(self.extraction_identifier)
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

            xml_file.delete()

            sample = PredictionSample(pdf_data=pdf_data, entity_name=entity_name, source_text=prediction_data.source_text)
            prediction_samples.append(sample)

        return prediction_samples

    def import_samples(self, for_training: bool) -> list[TrainingSample | PredictionSample]:
        samples: list[TrainingSample | PredictionSample] = list()
        max_retries = 3
        retry_delay = 5
        retries = 0

        url = f"{SERVICE_HOST}:{SERVICE_PORT}"
        url += "/get_samples_training" if for_training else "/get_samples_prediction"
        url += f"/{self.extraction_identifier.run_name}/{self.extraction_identifier.extraction_name}"

        while retries <= max_retries:
            try:
                response = requests.get(url)
                response.raise_for_status()

                if not response.json():
                    break

                samples.extend([TrainingSample(**x) if for_training else PredictionSample(**x) for x in response.json()])
                break
            except requests.exceptions.RequestException as e:
                config_logger.error(f"Error fetching training samples: {e}")
                retries += 1
                if retries <= max_retries:
                    config_logger.info(f"Retrying in {retry_delay} seconds... (attempt {retries}/{max_retries})")
                    sleep(retry_delay)
                else:
                    config_logger.error("Max retries reached. Exiting.")
                    break

        return samples

    def get_training_samples(self) -> list[TrainingSample]:
        key = SamplesCacheUseCase.get_training_cache_key(
            run_name=self.extraction_identifier.run_name, extraction_name=self.extraction_identifier.extraction_name
        )
        cached_samples = self.samples_cache_use_case.get_cached_samples(key)
        if cached_samples:
            return [TrainingSample(**sample) for sample in cached_samples]

        samples = self.import_samples(for_training=True)
        self.samples_cache_use_case.cache_samples(key, samples)
        return samples

    def get_prediction_samples_for_suggestions(self) -> list[PredictionSample]:
        return self.import_samples(for_training=False)

    def is_extractor_cancelled(self) -> bool:
        try:
            url = f"{SERVICE_HOST}:{SERVICE_PORT}"
            url += (
                f"/is_extractor_cancelled/{self.extraction_identifier.run_name}/{self.extraction_identifier.extraction_name}"
            )

            response = requests.get(url)
            response.raise_for_status()

            if response.status_code == 200:
                return response.json().get("cancelled", False)
            else:
                return False

        except requests.exceptions.RequestException as e:
            config_logger.error(f"Error checking if extractor is cancelled: {e}")
            return False

    def delete_cache(self):
        key = SamplesCacheUseCase.get_training_cache_key(
            run_name=self.extraction_identifier.run_name, extraction_name=self.extraction_identifier.extraction_name
        )
        self.samples_cache_use_case.delete_cache(key)
