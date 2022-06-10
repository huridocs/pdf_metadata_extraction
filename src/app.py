from typing import List, Dict
import json
import pymongo
from fastapi import FastAPI, HTTPException, UploadFile, File
import sys

from rsmq import RedisSMQ

from ServiceConfig import ServiceConfig
from data.LabeledData import LabeledData
from data.LabeledDataMultiOption import LabeledDataMultiOption
from data.PredictionData import PredictionData
from data.Suggestion import Suggestion
from data.SuggestionMultiOption import SuggestionMultiOption
from metadata_extraction.XmlFile import XmlFile

config = ServiceConfig()
logger = config.get_logger("service")

app = FastAPI()

logger.info(f"PDF information extraction service has started")


@app.get("/info")
async def info():
    logger.info("PDF information extraction endpoint")
    return sys.version


@app.get("/error")
async def error():
    logger.error("This is a test error from the error endpoint")
    raise HTTPException(status_code=500, detail="This is a test error from the error endpoint")


@app.post("/xml_to_train/{tenant}/{property_name}")
async def to_train_xml_file(tenant, property_name, file: UploadFile = File(...)):
    filename = '"No file name! Probably an error about the file in the request"'
    try:
        filename = file.filename
        xml_file = XmlFile(
            tenant=tenant,
            property_name=property_name,
            to_train=True,
            xml_file_name=filename,
        )
        xml_file.save(file=file.file.read())
        return "xml_to_train saved"
    except Exception:
        logger.error(f"Error adding task {filename}", exc_info=1)
        raise HTTPException(status_code=422, detail=f"Error adding task {filename}")


@app.post("/xml_to_predict/{tenant}/{property_name}")
async def to_predict_xml_file(tenant, property_name, file: UploadFile = File(...)):
    filename = '"No file name! Probably an error about the file in the request"'
    try:
        filename = file.filename
        xml_file = XmlFile(
            tenant=tenant,
            property_name=property_name,
            to_train=False,
            xml_file_name=filename,
        )
        xml_file.save(file=file.file.read())
        return "xml_to_train saved"
    except Exception:
        logger.error(f"Error adding task {filename}", exc_info=1)
        raise HTTPException(status_code=422, detail=f"Error adding task {filename}")


@app.get("/delete_queues")
async def delete_queues():
    try:
        results_queue = RedisSMQ(
            host=config.redis_host,
            port=config.redis_port,
            qname=config.results_queue_name,
        )

        results_queue.deleteQueue().execute()
        results_queue.createQueue().execute()

        return "deleted"
    except Exception:
        logger.error("Error", exc_info=1)
        raise HTTPException(status_code=422, detail="An error has occurred. Check graylog for more info")


@app.post("/labeled_data")
async def labeled_data_post(labeled_data: LabeledData):
    try:
        client = pymongo.MongoClient("mongodb://127.0.0.1:29017")
        pdf_metadata_extraction_db = client["pdf_information_extraction"]
        labeled_data = labeled_data.correct_data_scale()
        pdf_metadata_extraction_db.labeled_data.insert_one(labeled_data.dict())
        return "labeled data saved"
    except Exception:
        logger.error("Error", exc_info=1)
        raise HTTPException(status_code=422, detail="An error has occurred. Check graylog for more info")


@app.post("/labeled_data_multi_option")
async def labeled_data_multi_option_post(labeled_data: LabeledDataMultiOption):
    try:
        client = pymongo.MongoClient("mongodb://127.0.0.1:29017")
        pdf_metadata_extraction_db = client["pdf_information_extraction"]
        labeled_data = labeled_data.correct_data_scale()
        pdf_metadata_extraction_db.labeled_data_multi_option.insert_one(labeled_data.dict())
        return "labeled data saved"
    except Exception:
        logger.error("Error", exc_info=1)
        raise HTTPException(status_code=422, detail="An error has occurred. Check graylog for more info")


@app.post("/prediction_data")
async def prediction_data_post(prediction_data: PredictionData):
    try:
        client = pymongo.MongoClient("mongodb://127.0.0.1:29017")
        pdf_metadata_extraction_db = client["pdf_information_extraction"]
        pdf_metadata_extraction_db.predictiondata.insert_one(prediction_data.dict())
        return "prediction data saved"
    except Exception:
        logger.error("Error", exc_info=1)
        raise HTTPException(status_code=422, detail="An error has occurred. Check graylog for more info")


@app.get("/get_suggestions/{tenant}/{property_name}")
async def get_suggestions(tenant: str, property_name: str):
    try:
        client = pymongo.MongoClient("mongodb://127.0.0.1:29017")
        pdf_metadata_extraction_db = client["pdf_information_extraction"]
        suggestions_filter = {"tenant": tenant, "property_name": property_name}
        suggestions_list: List[Dict[str, str]] = list()

        for document in pdf_metadata_extraction_db.suggestions.find(suggestions_filter, no_cursor_timeout=True):
            suggestions_list.append(Suggestion(**document).dict())

        pdf_metadata_extraction_db.suggestions.delete_many(suggestions_filter)
        logger.info(f"{len(suggestions_list)} suggestions created for {tenant} {property_name}")
        return json.dumps(suggestions_list)
    except Exception:
        logger.error("Error", exc_info=1)
        raise HTTPException(status_code=422, detail="An error has occurred. Check graylog for more info")


@app.get("/get_suggestions_multi_option/{tenant}/{property_name}")
async def get_suggestions_multi_option(tenant: str, property_name: str):
    try:
        client = pymongo.MongoClient("mongodb://127.0.0.1:29017")
        pdf_metadata_extraction_db = client["pdf_information_extraction"]
        suggestions_filter = {"tenant": tenant, "property_name": property_name}
        suggestions_list: List[Dict[str, str]] = list()

        for document in pdf_metadata_extraction_db.suggestions_multi_option.find(suggestions_filter, no_cursor_timeout=True):
            suggestions_list.append(SuggestionMultiOption(**document).dict())

        pdf_metadata_extraction_db.suggestions.delete_many(suggestions_filter)
        logger.info(f"{len(suggestions_list)} suggestions multi option created for {tenant} {property_name}")
        return json.dumps(suggestions_list)
    except Exception:
        logger.error("Error", exc_info=1)
        raise HTTPException(status_code=422, detail="An error has occurred. Check graylog for more info")
