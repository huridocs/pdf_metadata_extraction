FROM ubuntu:20.04
USER root
RUN apt-get update

RUN mkdir /developer

WORKDIR /developer

RUN apt-get update && apt-get clean
RUN apt-get -y install python-dev python3-pip locales clang

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN pip3 install --upgrade pip

COPY . .

RUN pip3 install --default-timeout=200 -r requirements.txt

ENV FLASK_APP app.py

CMD python3 calculate_suggestions_async.py ;  gunicorn -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:5050
