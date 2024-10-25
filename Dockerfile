FROM pytorch/pytorch

RUN apt-get update && apt-get -y -q --no-install-recommends install libgomp1 pdftohtml
RUN apt-get -y install git
RUN mkdir -p /app/src /app/models_data

RUN addgroup --system python && adduser --system --group python
RUN chown -R python:python /app
USER python

ENV VIRTUAL_ENV=/app/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip --default-timeout=1000 install -r requirements.txt

WORKDIR /app
COPY ./src ./src

ENV PYTHONPATH "${PYTHONPATH}:/app/src"
ENV NLTK_DATA=/app/models_data/cache/nltk_data
ENV HF_DATASETS_CACHE=/app/models_data/cache/HF
ENV HF_HOME=/app/models_data/cache/HF_home
ENV TRANSFORMERS_CACHE=/app/models_data/cache/Transformers
ENV HF_HOME=/app/models_data/cache/Transformers
ENV TRANSFORMERS_VERBOSITY=error
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1
ENV CUDA_VISIBLE_DEVICES=0