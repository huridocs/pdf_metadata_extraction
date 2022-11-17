FROM python:3.9.12-bullseye AS base

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip --default-timeout=1000 install -r requirements.txt

RUN mkdir /app
RUN mkdir /app/src
WORKDIR /app
COPY ./src ./src

ENV NLTK_DATA=/app/docker_volume/cache/nltk_data
ENV HF_DATASETS_CACHE=/app/docker_volume/cache/HF
ENV HF_HOME=/app/docker_volume/cache/HF_home
ENV TRANSFORMERS_CACHE=/app/docker_volume/cache/Transformers

FROM base AS api
CMD gunicorn -k uvicorn.workers.UvicornWorker --chdir ./src app:app --bind 0.0.0.0:5052

FROM base AS queue_processor_information_extraction
RUN python3 src/metadata_extraction/PdfFeatures/embeddings_models.py
CMD python3 src/QueueProcessor.py