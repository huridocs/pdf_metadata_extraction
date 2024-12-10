import os
import shutil
from contextlib import asynccontextmanager
import json
from os.path import join

import pymongo
from catch_exceptions import catch_exceptions
from fastapi import FastAPI, HTTPException, UploadFile, File
import sys

from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
import sentry_sdk
from starlette.concurrency import run_in_threadpool
from trainable_entity_extractor.XmlFile import XmlFile
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.data.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.data.LabeledData import LabeledData
from trainable_entity_extractor.data.PredictionData import PredictionData
from trainable_entity_extractor.data.Suggestion import Suggestion
from trainable_entity_extractor.send_logs import send_logs

from Extractor import Extractor
from config import MONGO_HOST, MONGO_PORT, DATA_PATH
from data.ExtractionTask import ExtractionTask
from data.Options import Options
from data.Params import Params


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.mongodb_client = pymongo.MongoClient(f"{MONGO_HOST}:{MONGO_PORT}")
    yield
    app.mongodb_client.close()


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
            run_name=tenant, extraction_name=extraction_id, output_path=DATA_PATH
        ),
        to_train=True,
        xml_file_name=filename,
    )
    xml_file.save(file=file.file.read())
    return "xml_to_train saved"



@app.post("/xml_to_predict/{tenant}/{extraction_id}")
@catch_exceptions
async def to_predict_xml_file(tenant, extraction_id, file: UploadFile = File(...)):
    filename = file.filename
    xml_file = XmlFile(
        extraction_identifier=ExtractionIdentifier(
            run_name=tenant, extraction_name=extraction_id, output_path=DATA_PATH
        ),
        to_train=False,
        xml_file_name=filename,
    )
    xml_file.save(file=file.file.read())
    return "xml_to_train saved"



@app.post("/labeled_data")
@catch_exceptions
async def labeled_data_post(labeled_data: LabeledData):
    pdf_metadata_extraction_db = app.mongodb_client["pdf_metadata_extraction"]
    pdf_metadata_extraction_db.labeled_data.insert_one(labeled_data.scale_down_labels().to_dict())
    return "labeled data saved"


@app.post("/prediction_data")
@catch_exceptions
async def prediction_data_post(prediction_data: PredictionData):
    pdf_metadata_extraction_db = app.mongodb_client["pdf_metadata_extraction"]
    pdf_metadata_extraction_db.prediction_data.insert_one(prediction_data.to_dict())
    return "prediction data saved"


@app.get("/get_suggestions/{tenant}/{extraction_id}")
@catch_exceptions
async def get_suggestions(tenant: str, extraction_id: str):
    pdf_metadata_extraction_db = app.mongodb_client["pdf_metadata_extraction"]
    suggestions_filter = {"tenant": tenant, "id": extraction_id}
    suggestions_list: list[str] = list()

    for document in pdf_metadata_extraction_db.suggestions.find(suggestions_filter):
        suggestions_list.append(Suggestion(**document).scale_up().to_output())

    pdf_metadata_extraction_db.suggestions.delete_many(suggestions_filter)
    extraction_identifier = ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id, output_path=DATA_PATH)
    send_logs(extraction_identifier, f"{len(suggestions_list)} suggestions queried")

    return json.dumps(suggestions_list)


@app.delete("/{tenant}/{extraction_id}")
async def delete_model(tenant: str, extraction_id: str):
    shutil.rmtree(join(DATA_PATH, tenant, extraction_id), ignore_errors=True)
    return True


@app.get("/get_status/{tenant}/{extraction_id}")
async def get_satus(tenant: str, extraction_id: str):
    extraction_identifier = ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id, output_path=DATA_PATH)
    return extraction_identifier.get_status()

@app.post("/train/{tenant}/{extraction_id}")
@catch_exceptions
async def train(tenant: str, extraction_id: str):
    params = Params(id=extraction_id)
    task = ExtractionTask(tenant=tenant, task=Extractor.CREATE_MODEL_TASK_NAME, params=params)
    run_in_threadpool(Extractor.calculate_task, task)
    return True

@app.post("/predict/{tenant}/{extraction_id}")
@catch_exceptions
async def predict(tenant: str, extraction_id: str):
    params = Params(id=extraction_id)
    task = ExtractionTask(tenant=tenant, task=Extractor.SUGGESTIONS_TASK_NAME, params=params)
    run_in_threadpool(Extractor.calculate_task, task)
    return True

@app.post("/options")
@catch_exceptions
def save_options(options: Options):
    extraction_identifier = ExtractionIdentifier(
        run_name=options.tenant, extraction_name=options.extraction_id, output_path=DATA_PATH
    )
    extraction_identifier.save_options(options.options)
    os.utime(extraction_identifier.get_options_path().parent)
    config_logger.info(f"Options {options.options[:150]} saved for {extraction_identifier}")
    return True
