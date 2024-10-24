import os
from contextlib import asynccontextmanager
import json

import pymongo
from fastapi import FastAPI, HTTPException, UploadFile, File
import sys

from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
import sentry_sdk

from config import config_logger, MONGO_HOST, MONGO_PORT
from data.ExtractionIdentifier import ExtractionIdentifier
from data.LabeledData import LabeledData
from data.Option import Option
from data.Options import Options
from data.PredictionData import PredictionData
from data.Suggestion import Suggestion
from XmlFile import XmlFile
from send_logs import send_logs


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
async def to_train_xml_file(tenant, extraction_id, file: UploadFile = File(...)):
    filename = '"No file name! Probably an error about the file in the request"'
    try:
        filename = file.filename
        xml_file = XmlFile(
            extraction_identifier=ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id),
            to_train=True,
            xml_file_name=filename,
        )
        xml_file.save(file=file.file.read())
        return "xml_to_train saved"
    except Exception:
        config_logger.error(f"Error adding task {filename}", exc_info=1)
        raise HTTPException(status_code=422, detail=f"Error adding task {filename}")


@app.post("/xml_to_predict/{tenant}/{extraction_id}")
async def to_predict_xml_file(tenant, extraction_id, file: UploadFile = File(...)):
    filename = '"No file name! Probably an error about the file in the request"'
    try:
        filename = file.filename
        xml_file = XmlFile(
            extraction_identifier=ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id),
            to_train=False,
            xml_file_name=filename,
        )
        xml_file.save(file=file.file.read())
        return "xml_to_train saved"
    except Exception:
        config_logger.error(f"Error adding task {filename}", exc_info=1)
        raise HTTPException(status_code=422, detail=f"Error adding task {filename}")


@app.post("/labeled_data")
async def labeled_data_post(labeled_data: LabeledData):
    try:
        pdf_metadata_extraction_db = app.mongodb_client["pdf_metadata_extraction"]
        pdf_metadata_extraction_db.labeled_data.insert_one(labeled_data.scale_down_labels().to_dict())
        return "labeled data saved"
    except Exception:
        config_logger.error("Error", exc_info=1)
        raise HTTPException(status_code=422, detail="An error has occurred. Check graylog for more info")


@app.post("/prediction_data")
async def prediction_data_post(prediction_data: PredictionData):
    try:
        pdf_metadata_extraction_db = app.mongodb_client["pdf_metadata_extraction"]
        pdf_metadata_extraction_db.prediction_data.insert_one(prediction_data.to_dict())
        return "prediction data saved"
    except Exception:
        config_logger.error("Error", exc_info=1)
        raise HTTPException(status_code=422, detail="An error has occurred. Check graylog for more info")


@app.get("/get_suggestions/{tenant}/{extraction_id}")
async def get_suggestions(tenant: str, extraction_id: str):
    try:
        pdf_metadata_extraction_db = app.mongodb_client["pdf_metadata_extraction"]
        suggestions_filter = {"tenant": tenant, "id": extraction_id}
        suggestions_list: list[str] = list()

        for document in pdf_metadata_extraction_db.suggestions.find(suggestions_filter):
            suggestions_list.append(Suggestion(**document).scale_up().to_output())

        pdf_metadata_extraction_db.suggestions.delete_many(suggestions_filter)
        extraction_identifier = ExtractionIdentifier(run_name=tenant, extraction_name=extraction_id)
        send_logs(extraction_identifier, f"{len(suggestions_list)} suggestions queried")

        return json.dumps(suggestions_list)
    except Exception:
        config_logger.error("Error", exc_info=1)
        raise HTTPException(status_code=422, detail="An error has occurred. Check graylog for more info")


@app.post("/options")
def save_options(options: Options):
    try:
        extraction_identifier = ExtractionIdentifier(run_name=options.tenant, extraction_name=options.extraction_id)
        options_list = [option.model_dump() for option in options.options]
        extraction_identifier.get_options_path().write_text(json.dumps(options_list))
        os.utime(extraction_identifier.get_options_path().parent)
        config_logger.info(f"Options {options.options[:150]} saved for {extraction_identifier}")
        return True
    except Exception:
        config_logger.error("Error", exc_info=1)
        raise HTTPException(status_code=422, detail="An error has occurred. Check graylog for more info")
