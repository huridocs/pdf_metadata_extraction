import pymongo
from fastapi import FastAPI, HTTPException, UploadFile, File
import sys

from LabeledData import LabeledData
from get_graylog import get_graylog
from xml_file.XmlFile import XmlFile

graylog = get_graylog()

app = FastAPI()

graylog.info(f'PDF information extraction service has started')


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
    pdf_information_extraction_db.labeleddata.insert_one(labeled_data.dict())
    return 'labeled data saved'


@app.post('/labeled_xml/{tenant}')
async def labeled_xml(tenant, file: UploadFile = File(...)):
    filename = '"No file name! Probably an error about the file in the request"'
    try:
        filename = file.filename
        XmlFile(filename=filename, tenant=tenant).save(file=file.file.read())
        return 'task registered'
    except Exception:
        graylog.error(f'Error adding task {filename}', exc_info=1)
        raise HTTPException(status_code=422, detail=f'Error adding task {filename}')
