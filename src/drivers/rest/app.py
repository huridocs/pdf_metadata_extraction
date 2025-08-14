import os
import shutil
from contextlib import asynccontextmanager
import json

from queue_processor.QueueProcessor import QueueProcessor
from trainable_entity_extractor.domain.Suggestion import Suggestion
from trainable_entity_extractor.use_cases.XmlFile import XmlFile
from trainable_entity_extractor.use_cases.send_logs import send_logs

from adapters.MongoPersistenceRepository import MongoPersistenceRepository
from catch_exceptions import catch_exceptions
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import sys

from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
import sentry_sdk
from trainable_entity_extractor.config import config_logger, IS_TRAINING_CANCELED_FILE_NAME
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LabeledData import LabeledData
from trainable_entity_extractor.domain.PredictionData import PredictionData

from config import MODELS_DATA_PATH, REDIS_HOST, REDIS_PORT, PARAGRAPH_EXTRACTION_NAME
from domain.ParagraphExtractionData import ParagraphExtractionData
from domain.ParagraphExtractorTask import ParagraphExtractorTask
from domain.XML import XML
from drivers.rest.ParagraphsTranslations import ParagraphsTranslations
from use_cases.SampleProcessorUseCase import SampleProcessorUseCase


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.persistence_repository = MongoPersistenceRepository()
    yield
    app.persistence_repository.close()


app = FastAPI(lifespan=lifespan)

config_logger.info("PDF information extraction service has started")

try:
    sentry_sdk.init(
        os.environ.get("SENTRY_DSN"),
        traces_sample_rate=0.1,
        environment=os.environ.get("ENVIRONMENT", "development"),
    )
    app.add_middleware(SentryAsgiMiddleware)
except Exception:
    pass


@app.get("/")
@app.get("/info")
async def info():
    config_logger.info("PDF information extraction endpoint")
    return sys.version


@app.get("/error")
async def error():
    config_logger.error("This is a test error from the error endpoint")
    raise HTTPException(status_code=500, detail="This is a test error from the error endpoint")


@app.post("/xml_to_train/{tenant}/{extraction_id}")
@catch_exceptions
async def to_train_xml_file(tenant, extraction_id, file: UploadFile = File(...)):
    filename = file.filename
    xml_file = XmlFile(
        extraction_identifier=ExtractionIdentifier(
            run_name=tenant, extraction_name=extraction_id, output_path=MODELS_DATA_PATH
        ),
        to_train=True,
        xml_file_name=filename,
    )
    xml_file.save(file_content=file.file.read())
    return "xml_to_train saved"


@app.post("/xml_to_predict/{tenant}/{extraction_id}")
@catch_exceptions
async def to_predict_xml_file(tenant, extraction_id, file: UploadFile = File(...)):
    filename = file.filename
    xml_file = XmlFile(
        extraction_identifier=ExtractionIdentifier(
            run_name=tenant, extraction_name=extraction_id, output_path=MODELS_DATA_PATH
        ),
        to_train=False,
        xml_file_name=filename,
    )
    xml_file.save(file_content=file.file.read())
    return "xml_to_train saved"


@app.post("/labeled_data")
@catch_exceptions
async def labeled_data_post(labeled_data: LabeledData):
    labeled_data.scale_down_labels()
    extraction_identifier = ExtractionIdentifier(
        run_name=labeled_data.tenant, extraction_name=labeled_data.id, output_path=MODELS_DATA_PATH
    )
    app.persistence_repository.save_labeled_data(extraction_identifier, labeled_data)
    return "labeled data saved"


@app.get("/get_samples_training/{run_name}/{extraction_name}")
@catch_exceptions
async def get_samples_training(run_name: str, extraction_name: str):
    extraction_identifier = ExtractionIdentifier(
        run_name=run_name, extraction_name=extraction_name, output_path=MODELS_DATA_PATH
    )
    labeled_data = app.persistence_repository.load_and_delete_labeled_data(extraction_identifier, 50)
    return SampleProcessorUseCase.get_samples_for_training(
        extraction_identifier=extraction_identifier, labeled_data_list=labeled_data
    )


@app.get("/get_samples_prediction/{run_name}/{extraction_name}")
@catch_exceptions
async def get_samples_prediction(run_name: str, extraction_name: str):
    extraction_identifier = ExtractionIdentifier(
        run_name=run_name, extraction_name=extraction_name, output_path=MODELS_DATA_PATH
    )
    prediction_data = app.persistence_repository.load_and_delete_prediction_data(extraction_identifier, 50)
    return SampleProcessorUseCase.get_prediction_samples(
        extractor_identifier=extraction_identifier, prediction_data_list=prediction_data
    )


@app.post("/prediction_data")
@catch_exceptions
async def prediction_data_post(prediction_data: PredictionData):
    extraction_identifier = ExtractionIdentifier(
        run_name=prediction_data.tenant, extraction_name=prediction_data.id, output_path=MODELS_DATA_PATH
    )
    app.persistence_repository.save_prediction_data(extraction_identifier, prediction_data)
    return "prediction data saved"


@app.get("/get_suggestions/{run_name}/{extraction_name}")
@catch_exceptions
async def get_suggestions(run_name: str, extraction_name: str):
    extraction_identifier = ExtractionIdentifier(
        run_name=run_name, extraction_name=extraction_name, output_path=MODELS_DATA_PATH
    )
    suggestions = app.persistence_repository.load_suggestions(extraction_identifier)
    suggestions_list = [x.scale_up().to_output() for x in suggestions]
    send_logs(extraction_identifier, f"{len(suggestions_list)} suggestions queried")

    return json.dumps(suggestions_list)


@app.post("/save_suggestions/{run_name}/{extraction_name}")
@catch_exceptions
async def save_suggestions(run_name: str, extraction_name: str, suggestions: list[Suggestion]):
    extraction_identifier = ExtractionIdentifier(
        run_name=run_name, extraction_name=extraction_name, output_path=MODELS_DATA_PATH
    )
    app.persistence_repository.save_suggestions(extraction_identifier, suggestions)
    send_logs(extraction_identifier, f"{len(suggestions)} suggestions saved")
    return True


@app.post("/cancel_training/{run_name}/{extraction_name}")
async def cancel_training(run_name: str, extraction_name: str):
    extraction_identifier = ExtractionIdentifier(
        run_name=run_name, extraction_name=extraction_name, output_path=MODELS_DATA_PATH
    )
    app.persistence_repository.load_and_delete_labeled_data(extraction_identifier, 5000000)
    extraction_identifier.cancel_training()
    return True


@app.delete("/{run_name}/{extraction_name}")
async def delete_extractor(run_name: str, extraction_name: str):
    extraction_identifier = ExtractionIdentifier(
        run_name=run_name, extraction_name=extraction_name, output_path=MODELS_DATA_PATH
    )
    shutil.rmtree(extraction_identifier.get_path(), ignore_errors=True)
    app.persistence_repository.load_and_delete_labeled_data(extraction_identifier, 5000000)
    app.persistence_repository.load_and_delete_prediction_data(extraction_identifier, 5000000)
    return True


@app.post("/extract_paragraphs")
@catch_exceptions
async def extract_paragraphs(json_data: str = Form(...), xml_files: list[UploadFile] = File(...)):
    paragraph_extraction_data = ParagraphExtractionData(**json.loads(json_data))

    extractor_identifier = ExtractionIdentifier(
        run_name=PARAGRAPH_EXTRACTION_NAME, extraction_name=paragraph_extraction_data.key, output_path=MODELS_DATA_PATH
    )

    config_logger.info(f"extract_paragraphs endpoint called for {extractor_identifier.extraction_name}")
    app.persistence_repository.save_paragraph_extraction_data(extractor_identifier, paragraph_extraction_data)

    for file in xml_files:
        xml_file = XmlFile(
            extraction_identifier=extractor_identifier,
            to_train=True,
            xml_file_name=file.filename,
        )
        xml_file.save(file_content=file.file.read())

    paragraph_extractor_task = ParagraphExtractorTask(
        task=PARAGRAPH_EXTRACTION_NAME,
        key=paragraph_extraction_data.key,
        xmls=[XML(**x.model_dump()) for x in paragraph_extraction_data.xmls],
    )
    config_logger.info(f"add task {paragraph_extractor_task.model_dump()}")

    task = paragraph_extractor_task.model_dump()
    queue_name = paragraph_extraction_data.queue_name if paragraph_extraction_data.queue_name else PARAGRAPH_EXTRACTION_NAME
    QueueProcessor(REDIS_HOST, REDIS_PORT, [queue_name]).send_message(task)
    return "ok"


@app.get("/get_paragraphs_translations/{key}")
@catch_exceptions
async def get_paragraphs_translations(key: str) -> ParagraphsTranslations:
    extractor_identifier = ExtractionIdentifier(
        run_name=PARAGRAPH_EXTRACTION_NAME, extraction_name=key, output_path=MODELS_DATA_PATH
    )
    paragraphs_from_languages = app.persistence_repository.load_paragraphs_from_languages(extractor_identifier)
    return ParagraphsTranslations.from_paragraphs_from_languages(key, paragraphs_from_languages)
