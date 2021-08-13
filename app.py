import json
from typing import List, Dict

import pymongo
from fastapi import FastAPI, HTTPException, UploadFile, File
import sys

import threading
from data.LabeledData import LabeledData
from data.PredictionData import PredictionData
from data.Suggestion import Suggestion
from get_graylog import get_graylog
from information_extraction.InformationExtraction import InformationExtraction
from information_extraction.XmlFile import XmlFile

graylog = get_graylog()

app = FastAPI()

graylog.info(f'PDF information extraction service has started')


def sanitize_name(name: str):
    name = name.replace(' ', '_')
    name = ''.join(x for x in name if x.isalnum() or x == '_')
    return name


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
    client = pymongo.MongoClient('mongodb://mongo:27017')
    pdf_information_extraction_db = client['pdf_information_extraction']
    labeled_data.tenant = sanitize_name(labeled_data.tenant)
    labeled_data.extraction_name = sanitize_name(labeled_data.extraction_name)
    pdf_information_extraction_db.labeleddata.insert_one(labeled_data.dict())
    return 'labeled data saved'


@app.post('/prediction_data')
async def prediction_data_post(prediction_data: PredictionData):
    client = pymongo.MongoClient('mongodb://mongo:27017')
    pdf_information_extraction_db = client['pdf_information_extraction']
    prediction_data.tenant = sanitize_name(prediction_data.tenant)
    prediction_data.extraction_name = sanitize_name(prediction_data.extraction_name)
    pdf_information_extraction_db.predictiondata.insert_one(prediction_data.dict())
    return 'labeled data saved'


@app.post('/xml_file/{tenant}/{extraction_name}')
async def xml_file(tenant, extraction_name, file: UploadFile = File(...)):
    filename = '"No file name! Probably an error about the file in the request"'
    try:
        filename = file.filename
        tenant = sanitize_name(tenant)
        extraction_name = sanitize_name(extraction_name)
        XmlFile(file_name=filename, tenant=tenant, extraction_name=extraction_name).save(file=file.file.read())
        return 'task registered'
    except Exception:
        graylog.error(f'Error adding task {filename}', exc_info=1)
        raise HTTPException(status_code=422, detail=f'Error adding task {filename}')


@app.get('/get_suggestions/{tenant}/{extraction_name}')
async def get_suggestions(tenant: str, extraction_name: str):
    tenant = sanitize_name(tenant)
    extraction_name = sanitize_name(extraction_name)
    client = pymongo.MongoClient('mongodb://mongo:27017')
    pdf_information_extraction_db = client['pdf_information_extraction']
    find_filter = {"extraction_name": extraction_name, "tenant": tenant}
    suggestions_list: List[Dict[str, str]] = list()

    for document in pdf_information_extraction_db.suggestions.find(find_filter, no_cursor_timeout=True):
        suggestions_list.append(Suggestion(**document).dict())

    return json.dumps(suggestions_list)


@app.post('/calculate_suggestions/{tenant}/{extraction_name}')
async def calculate_suggestions(tenant: str, extraction_name: str):
    tenant = sanitize_name(tenant)
    extraction_name = sanitize_name(extraction_name)
    information_extraction = InformationExtraction(tenant=tenant, extraction_name=extraction_name)
    information_extraction.calculate_suggestions()
    return 'Started'

