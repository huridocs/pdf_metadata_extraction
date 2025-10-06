from datetime import timedelta, datetime
from pathlib import Path
from time import sleep
import shutil
import requests
from ml_cloud_connector.adapters.google_v2.GoogleCloudStorage import GoogleCloudStorage
from ml_cloud_connector.domain.ServerParameters import ServerParameters
from ml_cloud_connector.domain.ServerType import ServerType
from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Performance import Performance
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.use_cases.PredictUseCase import PredictUseCase
from trainable_entity_extractor.use_cases.TrainUseCase import TrainUseCase

from adapters.CloudModelStorage import CloudModelStorage
from config import SERVICE_HOST, SERVICE_PORT, MODELS_DATA_PATH
from drivers.extractors import EXTRACTORS
from use_cases.SampleProcessorUseCase import SampleProcessorUseCase


logger = ExtractorLogger()
train_use_case = TrainUseCase(EXTRACTORS, logger)
predict_use_case = PredictUseCase(EXTRACTORS, logger)

google_cloud_storage = None

if GoogleCloudStorage.could_be_configured():
    server_parameters = ServerParameters(namespace="metadata_extractor", server_type=ServerType.METADATA_EXTRACTION)
    google_cloud_storage = GoogleCloudStorage(server_parameters, config_logger)
    config_logger.info("Google Cloud Storage client initialized successfully")

cloud_storage = CloudModelStorage(google_cloud_storage, logger)


def ensure_fresh_model_folder(extraction_identifier: ExtractionIdentifier, max_age_hours: int = 1) -> None:
    path = Path(extraction_identifier.get_path())

    if path.exists():
        folder_modified_time = datetime.fromtimestamp(path.stat().st_mtime)
        current_time = datetime.now()
        age = current_time - folder_modified_time

        if age > timedelta(hours=max_age_hours):
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


def performance_one_method(extractor_job: TrainableEntityExtractorJob) -> Performance:
    extraction_identifier = ExtractionIdentifier(
        run_name=extractor_job.run_name, output_path=MODELS_DATA_PATH, extraction_name=extractor_job.extraction_name
    )

    ensure_fresh_model_folder(extraction_identifier)

    sample_processor = SampleProcessorUseCase(extraction_identifier)
    samples = sample_processor.get_training_samples()
    extraction_data = ExtractionData(
        samples=samples,
        options=extractor_job.options,
        multi_value=extractor_job.multi_value,
        extraction_identifier=extraction_identifier,
    )
    return train_use_case.get_performance(extractor_job, extraction_data)


def train_one_method(extractor_job: TrainableEntityExtractorJob) -> bool:
    extraction_identifier = ExtractionIdentifier(
        run_name=extractor_job.run_name, output_path=MODELS_DATA_PATH, extraction_name=extractor_job.extraction_name
    )
    sample_processor = SampleProcessorUseCase(extraction_identifier)
    samples = sample_processor.get_training_samples()
    sample_processor.delete_cache()
    extraction_data = ExtractionData(
        samples=samples,
        options=extractor_job.options,
        multi_value=extractor_job.multi_value,
        extraction_identifier=extraction_identifier,
    )
    success, message = train_use_case.train_one_method(extractor_job, extraction_data)
    extraction_identifier.clean_extractor_folder(extractor_job.method_name)

    if not success:
        return False

    return cloud_storage.upload_model(extraction_identifier, extractor_job)


def distributed_predict(extractor_job: TrainableEntityExtractorJob) -> bool:
    extraction_identifier = ExtractionIdentifier(
        run_name=extractor_job.run_name, output_path=MODELS_DATA_PATH, extraction_name=extractor_job.extraction_name
    )

    success = cloud_storage.download_model(extraction_identifier)
    if not success:
        return False

    sample_processor = SampleProcessorUseCase(extraction_identifier)
    samples = sample_processor.get_prediction_samples_for_suggestions()
    extractor_job = extractor_job.set_extractors_path(MODELS_DATA_PATH)
    suggestions = predict_use_case.predict(extractor_job, samples)
    return _send_suggestions(extraction_identifier, suggestions)[0]


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
