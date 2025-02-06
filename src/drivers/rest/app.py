import os
import shutil
from contextlib import asynccontextmanager
import json
from os.path import join

from queue_processor.QueueProcessor import QueueProcessor

from adapters.MongoPersistenceRepository import MongoPersistenceRepository
from catch_exceptions import catch_exceptions
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import sys

from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
import sentry_sdk
from trainable_entity_extractor.XmlFile import XmlFile
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.PredictionData import PredictionData
from trainable_entity_extractor.send_logs import send_logs

from config import DATA_PATH, REDIS_HOST, REDIS_PORT, PARAGRAPH_EXTRACTION_NAME
from domain.ParagraphExtractionData import ParagraphExtractionData
from domain.ParagraphExtractorTask import ParagraphExtractorTask
from domain.XML import XML
from drivers.rest.ParagraphsTranslations import ParagraphsTranslations


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
        extraction_identifier=ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id, output_path=DATA_PATH),
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
        extraction_identifier=ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id, output_path=DATA_PATH),
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
        run_name=labeled_data.tenant, extraction_name=labeled_data.id, output_path=DATA_PATH
    )
    app.persistence_repository.save_labeled_data(extraction_identifier, labeled_data)
    return "labeled data saved"


@app.post("/prediction_data")
@catch_exceptions
async def prediction_data_post(prediction_data: PredictionData):
    extraction_identifier = ExtractionIdentifier(
        run_name=prediction_data.tenant, extraction_name=prediction_data.id, output_path=DATA_PATH
    )
    app.persistence_repository.save_prediction_data(extraction_identifier, prediction_data)
    return "prediction data saved"


@app.get("/get_suggestions/{run_name}/{extraction_name}")
@catch_exceptions
async def get_suggestions(run_name: str, extraction_name: str):
    extraction_identifier = ExtractionIdentifier(run_name=run_name, extraction_name=extraction_name, output_path=DATA_PATH)
    suggestions = app.persistence_repository.load_suggestions(extraction_identifier)
    suggestions_list = [x.scale_up().to_output() for x in suggestions]
    send_logs(extraction_identifier, f"{len(suggestions_list)} suggestions queried")

    return json.dumps(suggestions_list)


@app.delete("/{run_name}/{extraction_name}")
async def remove_extractor(run_name: str, extraction_name: str):
    shutil.rmtree(join(DATA_PATH, run_name, extraction_name), ignore_errors=True)
    return True


@app.post("/extract_paragraphs")
@catch_exceptions
async def extract_paragraphs(json_data: str = Form(...), xml_files: list[UploadFile] = File(...)):
    paragraph_extraction_data = ParagraphExtractionData(**json.loads(json_data))

    extractor_identifier = ExtractionIdentifier(
        run_name=PARAGRAPH_EXTRACTION_NAME, extraction_name=paragraph_extraction_data.key, output_path=DATA_PATH
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
        xmls=[XML(**x.model_dump()) for x in paragraph_extraction_data.xmls_segments],
    )
    config_logger.info(f"add task {paragraph_extractor_task.model_dump()}")

    task = paragraph_extractor_task.model_dump()
    QueueProcessor(REDIS_HOST, REDIS_PORT, [PARAGRAPH_EXTRACTION_NAME]).send_message(task)
    return "ok"


@app.get("/get_paragraphs_translations/{key}")
@catch_exceptions
async def get_paragraphs_translations(key: str) -> ParagraphsTranslations:
    extractor_identifier = ExtractionIdentifier(
        run_name=PARAGRAPH_EXTRACTION_NAME, extraction_name=key, output_path=DATA_PATH
    )
    paragraphs_from_languages = app.persistence_repository.load_paragraphs_from_languages(extractor_identifier)
    app.persistence_repository.delete_paragraphs_from_languages(extractor_identifier)
    return ParagraphsTranslations.from_paragraphs_from_languages(key, paragraphs_from_languages)
