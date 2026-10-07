FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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
RUN uv pip install --upgrade pip
RUN uv pip install -r requirements.txt

WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:/app/src"
ENV NLTK_DATA=/app/models_data/cache/nltk_data
ENV HF_DATASETS_CACHE=/app/models_data/cache/HF
ENV HF_HOME=/app/models_data/cache/HF_home
ENV FLAIR_CACHE_ROOT=/app/flair_cache
ENV TRANSFORMERS_VERBOSITY=error
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1
ENV CUDA_VISIBLE_DEVICES=0

# The Flair NER text-to-text methods are disabled (see FLAIR_NER_METHODS_ENABLED in src/config.py),
# so the ~2.2 GB `ner-ontonotes-large` checkpoint is not needed by default. Build with
# `--build-arg INSTALL_FLAIR_MODELS=true` (and run with FLAIR_NER_METHODS_ENABLED=true) to install
# it again once the Flair/torch loading issue is resolved.
ARG INSTALL_FLAIR_MODELS=false
COPY ./scripts ./scripts
RUN if [ "$INSTALL_FLAIR_MODELS" = "true" ]; then \
        python scripts/download_flair_models.py; \
    else \
        echo "Skipping Flair NER model download (Flair NER methods disabled)"; \
    fi

COPY ./src ./src

