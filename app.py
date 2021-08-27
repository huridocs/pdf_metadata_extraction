from typing import List, Dict
import json
import pymongo
from fastapi import FastAPI, HTTPException, UploadFile, File
import sys

from data.LabeledData import LabeledData
from data.PredictionData import PredictionData
from data.Suggestion import Suggestion
from data.CreateModelTask import CreateModelTask
from get_logger import get_logger
from information_extraction.XmlFile import XmlFile

logger = get_logger()

app = FastAPI()

logger.info(f'PDF information extraction service has started')


@app.get('/info')
async def info():
    logger.info('PDF information extraction endpoint')
    return sys.version


@app.get('/error')
async def error():
    logger.error("This is a test error from the error endpoint")
    raise HTTPException(status_code=500, detail='This is a test error from the error endpoint')


@app.post('/to_train_xml/{tenant}/{template}/{property_name}')
async def training_xml_file(tenant, template, property_name, file: UploadFile = File(...)):
    filename = '"No file name! Probably an error about the file in the request"'
    try:
        filename = file.filename
        xml_file = XmlFile(template=template, tenant=tenant, property_name=property_name, file_name=filename)
        xml_file.save_as_to_train(file=file.file.read())
        return 'xml saved'
    except Exception:
        logger.error(f'Error adding task {filename}', exc_info=1)
        raise HTTPException(status_code=422, detail=f'Error adding task {filename}')


@app.post('/to_predict_xml/{tenant}/{template}/{property_name}')
async def training_xml_file(tenant, template, property_name, file: UploadFile = File(...)):
    filename = '"No file name! Probably an error about the file in the request"'
    try:
        filename = file.filename
        xml_file = XmlFile(template=template, tenant=tenant, property_name=property_name, file_name=filename)
        xml_file.save_as_to_predict(file=file.file.read())
        return 'xml saved'
    except Exception:
        logger.error(f'Error adding task {filename}', exc_info=1)
        raise HTTPException(status_code=422, detail=f'Error adding task {filename}')


@app.post('/labeled_data')
async def labeled_data_post(labeled_data: LabeledData):
    try:
        client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')
        pdf_information_extraction_db = client['pdf_information_extraction']
        pdf_information_extraction_db.labeleddata.insert_one(labeled_data.dict())
        return 'labeled data saved'
    except Exception:
        logger.error('Error', exc_info=1)
        raise HTTPException(status_code=422, detail='An error has occurred. Check graylog for more info')


@app.post('/prediction_data')
async def prediction_data_post(prediction_data: PredictionData):
    try:
        client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')
        pdf_information_extraction_db = client['pdf_information_extraction']
        pdf_information_extraction_db.predictiondata.insert_one(prediction_data.dict())
        return 'prediction data saved'
    except Exception:
        logger.error('Error', exc_info=1)
        raise HTTPException(status_code=422, detail='An error has occurred. Check graylog for more info')


@app.post('/create_model')
async def create_model_post(task: CreateModelTask):
    try:
        client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')
        pdf_information_extraction_db = client['pdf_information_extraction']
        pdf_information_extraction_db.tasks.delete_many(task.dict())
        pdf_information_extraction_db.tasks.insert_one(task.dict())
        return 'task added'
    except Exception:
        logger.error('Error', exc_info=1)
        raise HTTPException(status_code=422, detail='An error has occurred. Check graylog for more info')


@app.get('/get_suggestions/{tenant}/{template}/{property_name}')
async def get_suggestions(tenant: str, template: str, property_name: str):
    try:
        client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')
        pdf_information_extraction_db = client['pdf_information_extraction']
        suggestions_filter = {"tenant": tenant, "template": template, "property_name": property_name}
        suggestions_list: List[Dict[str, str]] = list()

        for document in pdf_information_extraction_db.suggestions.find(suggestions_filter, no_cursor_timeout=True):
            suggestions_list.append(Suggestion(**document).dict())

        pdf_information_extraction_db.suggestions.delete_many(suggestions_filter)
        logger.info(f'{len(suggestions_list)} suggestions created for {tenant} {property_name}')
        return json.dumps(suggestions_list)
    except Exception:
        logger.error('Error', exc_info=1)
        raise HTTPException(status_code=422, detail='An error has occurred. Check graylog for more info')