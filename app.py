from fastapi import FastAPI, HTTPException
import sys
from get_graylog import get_graylog

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
async def labeled_data():
    pass