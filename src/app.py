from typing import List, Dict
import json
import pymongo
from fastapi import FastAPI, HTTPException, UploadFile, File
import sys

from ServiceConfig import ServiceConfig
from data.LabeledData import LabeledData
from data.PredictionData import PredictionData
from data.Suggestion import Suggestion
from information_extraction.XmlFile import XmlFile

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
    raise HTTPException(
        status_code=500, detail="This is a test error from the error endpoint"
    )


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


@app.post("/labeled_data")
async def labeled_data_post(labeled_data: LabeledData):
    try:
        client = pymongo.MongoClient("mongodb://127.0.0.1:27017")
        pdf_information_extraction_db = client["pdf_information_extraction"]
        pdf_information_extraction_db.labeleddata.insert_one(labeled_data.dict())
        return "labeled data saved"
    except Exception:
        logger.error("Error", exc_info=1)
        raise HTTPException(
            status_code=422, detail="An error has occurred. Check graylog for more info"
        )


@app.post("/prediction_data")
async def prediction_data_post(prediction_data: PredictionData):
    try:
        client = pymongo.MongoClient("mongodb://127.0.0.1:27017")
        pdf_information_extraction_db = client["pdf_information_extraction"]
        pdf_information_extraction_db.predictiondata.insert_one(prediction_data.dict())
        return "prediction data saved"
    except Exception:
        logger.error("Error", exc_info=1)
        raise HTTPException(
            status_code=422, detail="An error has occurred. Check graylog for more info"
        )


@app.get("/get_suggestions/{tenant}/{property_name}")
async def get_suggestions(tenant: str, property_name: str):
    try:
        client = pymongo.MongoClient("mongodb://127.0.0.1:27017")
        pdf_information_extraction_db = client["pdf_information_extraction"]
        suggestions_filter = {"tenant": tenant, "property_name": property_name}
        suggestions_list: List[Dict[str, str]] = list()

        for document in pdf_information_extraction_db.suggestions.find(
            suggestions_filter, no_cursor_timeout=True
        ):
            suggestions_list.append(Suggestion(**document).dict())

        pdf_information_extraction_db.suggestions.delete_many(suggestions_filter)
        logger.info(
            f"{len(suggestions_list)} suggestions created for {tenant} {property_name}"
        )
        return json.dumps(suggestions_list)
    except Exception:
        logger.error("Error", exc_info=1)
        raise HTTPException(
            status_code=422, detail="An error has occurred. Check graylog for more info"
        )
