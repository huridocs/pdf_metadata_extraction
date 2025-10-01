from time import sleep

import requests
from ml_cloud_connector.adapters.google_v2.GoogleCloudStorage import GoogleCloudStorage
from ml_cloud_connector.domain.ServerParameters import ServerParameters
from ml_cloud_connector.domain.ServerType import ServerType
from trainable_entity_extractor.adapters.ExtractorLogger import ExtractorLogger
from trainable_entity_extractor.adapters.extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import (
    PdfToMultiOptionExtractor,
)
from trainable_entity_extractor.adapters.extractors.pdf_to_text_extractor.PdfToTextExtractor import PdfToTextExtractor
from trainable_entity_extractor.adapters.extractors.text_to_multi_option_extractor.TextToMultiOptionExtractor import (
    TextToMultiOptionExtractor,
)
from trainable_entity_extractor.adapters.extractors.text_to_text_extractor.TextToTextExtractor import TextToTextExtractor
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.domain.Performance import Performance
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase
from trainable_entity_extractor.use_cases.PredictUseCase import PredictUseCase
from trainable_entity_extractor.use_cases.TrainUseCase import TrainUseCase

from adapters.CloudModelStorage import CloudModelStorage
from config import DATA_PATH, SERVICE_HOST, SERVICE_PORT
from use_cases.SampleProcessorUseCase import SampleProcessorUseCase

EXTRACTORS: list[type[ExtractorBase]] = [
    PdfToMultiOptionExtractor,
    TextToMultiOptionExtractor,
    PdfToTextExtractor,
    TextToTextExtractor,
]

logger = ExtractorLogger()
train_use_case = TrainUseCase(EXTRACTORS, logger)
predict_use_case = PredictUseCase(EXTRACTORS, logger)

try:
    server_parameters = ServerParameters(namespace="metadata_extractor", server_type=ServerType.METADATA_EXTRACTION)
    google_cloud_storage = GoogleCloudStorage(server_parameters, config_logger)
    if not google_cloud_storage.is_properly_configured():
        google_cloud_storage = None
    config_logger.info("Google Cloud Storage client initialized successfully")
except Exception as e:
    config_logger.error(f"Failed to initialize Google Cloud Storage client: {e}")
    google_cloud_storage = None

cloud_storage = CloudModelStorage(google_cloud_storage, logger)


def performance_one_method(
    extractor_job: TrainableEntityExtractorJob, options: list[Option], multi_value: bool
) -> Performance:
    extraction_identifier = ExtractionIdentifier(
        run_name=extractor_job.run_name, output_path=DATA_PATH, extraction_name=extractor_job.extraction_name
    )
    sample_processor = SampleProcessorUseCase(extraction_identifier)
    samples = sample_processor.get_training_samples()
    extraction_data = ExtractionData(
        samples=samples,
        options=options,
        multi_value=multi_value,
        extraction_identifier=extraction_identifier,
    )
    return train_use_case.get_performance(extractor_job, extraction_data)


def train_one_method(
    extractor_job: TrainableEntityExtractorJob, options: list[Option], multi_value: bool
) -> tuple[bool, str]:
    extraction_identifier = ExtractionIdentifier(
        run_name=extractor_job.run_name, output_path=DATA_PATH, extraction_name=extractor_job.extraction_name
    )
    sample_processor = SampleProcessorUseCase(extraction_identifier)
    samples = sample_processor.get_training_samples()
    extraction_data = ExtractionData(
        samples=samples,
        options=options,
        multi_value=multi_value,
        extraction_identifier=extraction_identifier,
    )
    success, message = train_use_case.train_one_method(extractor_job, extraction_data)
    return success, message


def distributed_predict(extractor_job: TrainableEntityExtractorJob) -> tuple[bool, str]:
    extraction_identifier = ExtractionIdentifier(
        run_name=extractor_job.run_name, output_path=DATA_PATH, extraction_name=extractor_job.extraction_name
    )

    success = cloud_storage.download_model(extraction_identifier)
    if not success:
        return False, "Could not download model from cloud storage"

    sample_processor = SampleProcessorUseCase(extraction_identifier)
    samples = sample_processor.get_prediction_samples_for_suggestions()
    suggestions = predict_use_case.predict(extractor_job, samples)
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
