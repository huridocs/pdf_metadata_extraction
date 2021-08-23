from typing import List, Dict
import json
import pymongo
from fastapi import FastAPI, HTTPException, UploadFile, File
import sys

from data.LabeledData import LabeledData
from data.PredictionData import PredictionData
from data.Suggestion import Suggestion
from data.Task import Task
from get_graylog import get_graylog
from information_extraction.InformationExtraction import InformationExtraction
from information_extraction.XmlFile import XmlFile

graylog = get_graylog()

app = FastAPI()

graylog.info(f'PDF information extraction service has started')


def sanitize_name(name: str):
    return ''.join(x if x.isalnum() else '_' for x in name)


@app.get('/info')
async def info():
    graylog.info('PDF information extraction endpoint')
    return sys.version


@app.get('/error')
async def error():
    graylog.error("This is a test error from the error endpoint")
    raise HTTPException(status_code=500, detail='This is a test error from the error endpoint')


@app.post('/labeled_data')
async def labeled_data_post(labeled_data: LabeledData):
    try:
        client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')
        pdf_information_extraction_db = client['pdf_information_extraction']
        labeled_data.tenant = sanitize_name(labeled_data.tenant)
        labeled_data.extraction_name = sanitize_name(labeled_data.extraction_name)
        pdf_information_extraction_db.labeleddata.insert_one(labeled_data.dict())
        return 'labeled data saved'
    except Exception:
        graylog.error('Error', exc_info=1)
        raise HTTPException(status_code=422, detail='An error has occurred. Check graylog for more info')


@app.post('/prediction_data')
async def prediction_data_post(prediction_data: PredictionData):
    try:
        client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')
        pdf_information_extraction_db = client['pdf_information_extraction']
        prediction_data.tenant = sanitize_name(prediction_data.tenant)
        prediction_data.extraction_name = sanitize_name(prediction_data.extraction_name)
        pdf_information_extraction_db.predictiondata.insert_one(prediction_data.dict())
        return 'prediction data saved'
    except Exception:
        graylog.error('Error', exc_info=1)
        raise HTTPException(status_code=422, detail='An error has occurred. Check graylog for more info')


@app.post('/add_task')
async def prediction_data_post(task: Task):
    try:
        client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')
        pdf_information_extraction_db = client['pdf_information_extraction']
        task.tenant = sanitize_name(task.tenant)
        task.extraction_name = sanitize_name(task.extraction_name)
        pdf_information_extraction_db.tasks.delete_many(task.dict())
        pdf_information_extraction_db.tasks.insert_one(task.dict())
        return 'task added'
    except Exception:
        graylog.error('Error', exc_info=1)
        raise HTTPException(status_code=422, detail='An error has occurred. Check graylog for more info')


@app.post('/xml_file/{tenant}/{extraction_name}')
async def xml_file(tenant, extraction_name, file: UploadFile = File(...)):
    filename = '"No file name! Probably an error about the file in the request"'
    try:
        filename = file.filename
        tenant = sanitize_name(tenant)
        extraction_name = sanitize_name(extraction_name)
        XmlFile(file_name=filename, tenant=tenant, extraction_name=extraction_name).save(file=file.file.read())
        return 'xml saved'
    except Exception:
        graylog.error(f'Error adding task {filename}', exc_info=1)
        raise HTTPException(status_code=422, detail=f'Error adding task {filename}')


@app.get('/get_suggestions/{tenant}/{extraction_name}')
async def get_suggestions(tenant: str, extraction_name: str):
    tenant = sanitize_name(tenant)
    extraction_name = sanitize_name(extraction_name)
    try:
        client = pymongo.MongoClient('mongodb://mongo_information_extraction:27017')
        pdf_information_extraction_db = client['pdf_information_extraction']
        suggestions_filter = {"extraction_name": extraction_name, "tenant": tenant}
        suggestions_list: List[Dict[str, str]] = list()

        for document in pdf_information_extraction_db.suggestions.find(suggestions_filter, no_cursor_timeout=True):
            suggestions_list.append(Suggestion(**document).dict())

        pdf_information_extraction_db.suggestions.delete_many(suggestions_filter)
        graylog.info(f'{len(suggestions_list)} suggestions created for {tenant} {extraction_name}')
        return json.dumps(suggestions_list)
    except Exception:
        graylog.error('Error', exc_info=1)
        raise HTTPException(status_code=422, detail='An error has occurred. Check graylog for more info')


@app.post('/calculate_suggestions/{tenant}/{extraction_name}')
async def calculate_suggestions(tenant: str, extraction_name: str):
    try:
        tenant = sanitize_name(tenant)
        extraction_name = sanitize_name(extraction_name)
        information_extraction = InformationExtraction(tenant=tenant, extraction_name=extraction_name)
        information_extraction.calculate_suggestions()
        return 'Started'
    except Exception:
        graylog.error('Error', exc_info=1)
        raise HTTPException(status_code=422, detail='An error has occurred. Check graylog for more info')

