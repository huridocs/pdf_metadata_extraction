from time import sleep
import os

import requests
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.Performance import Performance
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.use_cases.TrainableEntityExtractor import TrainableEntityExtractor

from config import DATA_PATH, SERVICE_HOST, SERVICE_PORT
from drivers.distributed_worker.model_to_cloud import upload_model_to_cloud, download_model_from_cloud, check_model_completion_signal
from use_cases.SampleProcessorUseCase import SampleProcessorUseCase


def performance_one_method(
    extractor_job: TrainableEntityExtractorJob, options: list[Option], multi_value: bool
) -> Performance:
    extraction_identifier = ExtractionIdentifier(
        run_name=extractor_job.run_name, output_path=DATA_PATH, extraction_name=extractor_job.extraction_name
    )
    sample_processor = SampleProcessorUseCase(extraction_identifier)
    samples = sample_processor.get_training_samples()
    trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier)
    extraction_data = ExtractionData(
        samples=samples,
        options=options,
        multi_value=multi_value,
        extraction_identifier=extraction_identifier,
    )
    return trainable_entity_extractor.get_performance(extractor_job, extraction_data)


def train_one_method(
    extractor_job: TrainableEntityExtractorJob, options: list[Option], multi_value: bool
) -> tuple[bool, str]:
    extraction_identifier = ExtractionIdentifier(
        run_name=extractor_job.run_name, output_path=DATA_PATH, extraction_name=extractor_job.extraction_name
    )
    sample_processor = SampleProcessorUseCase(extraction_identifier)
    samples = sample_processor.get_training_samples()
    trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier)
    extraction_data = ExtractionData(
        samples=samples,
        options=options,
        multi_value=multi_value,
        extraction_identifier=extraction_identifier,
    )
    success, message = trainable_entity_extractor.train_one_method(extractor_job, extraction_data)
    if success:
        upload_model_to_cloud(extraction_identifier, extractor_job.run_name)
    return success, message


def distributed_predict(extractor_job: TrainableEntityExtractorJob) -> tuple[bool, str]:
    extraction_identifier = ExtractionIdentifier(
        run_name=extractor_job.run_name, output_path=DATA_PATH, extraction_name=extractor_job.extraction_name
    )
    model_path = extraction_identifier.get_path()

    if not os.path.exists(model_path):
        max_retries = 10
        base_delay = 30
        max_delay = 15*60

        for attempt in range(max_retries + 1):
            if check_model_completion_signal(extraction_identifier):
                success = download_model_from_cloud(extraction_identifier)
                if success:
                    break
                else:
                    return False, f"Model not found locally and could not be downloaded from cloud: {model_path}"
            else:
                if attempt == max_retries:
                    return False, f"Model upload not yet complete for {model_path}. Completion signal not found after {max_retries} retries."

                delay = min(base_delay * (2 ** attempt), max_delay)
                config_logger.info(f"Model upload not complete for {model_path}. Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                sleep(delay)

    sample_processor = SampleProcessorUseCase(extraction_identifier)
    samples = sample_processor.get_prediction_samples_for_suggestions()
    trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier)
    suggestions = trainable_entity_extractor.predict(samples)
    return _send_suggestions(extraction_identifier, suggestions)


def _send_suggestions(extraction_identifier: ExtractionIdentifier, suggestions: list[Suggestion]) -> tuple[bool, str]:
    max_retries = 3
    retry_delay = 5  # seconds

    url = f"{SERVICE_HOST}:{SERVICE_PORT}"
    url += "/save_suggestions"
    url += f"/{extraction_identifier.run_name}/{extraction_identifier.extraction_name}"
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    json_data = [x.model_dump() for x in suggestions]

    while True:
        try:
            response = requests.post(url, headers=headers, json=json_data)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            config_logger.error(f"Error fetching training samples: {e}")
            if max_retries > 0:
                max_retries -= 1
                config_logger.info(f"Retrying in {retry_delay} seconds...")
                sleep(retry_delay)
                continue
            else:
                config_logger.error("Max retries reached. Exiting.")
                return False, "Could not send suggestions back"

    return True, ""
