<h3 align="center">PDF metadata extraction</h3>
<p align="center">A Docker-powered service for extracting pieces of information from PDFs</p>

---

## Contents

- [Dependencies](#dependencies)
- [Requirements](#requirements)
- [Docker containers](#docker-containers)
- [How to use it](#how-to-use-it)
- [HTTP server](#http-server)
- [Queue processor](#queue-processor)
- [Service configuration](#service-configuration)
- [Get service logs](#get-service-logs)
- [Set up environment for development](#set-up-environment-for-development)
- [Execute tests](#execute-tests)
- [Troubleshooting](#troubleshooting)

## Dependencies
* Docker 20.10.14 [install link](https://runnable.com/docker/getting-started/)
* Docker-compose 2.4.1 

    Note: On mac Docker-compose is installed with Docker

    [install link](https://docs.docker.com/compose/install/) 

    [install on Ubuntu link](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-compose-on-ubuntu-20-04)

## Requirements

* 12Gb RAM memory
* Single core

## Docker containers

A redis server is needed to use the service asynchronously. For that matter, it can be used the 
command `./run start:testing` that has a built-in 
redis server.

Containers with `./run start`

![Alt logo](readme_pictures/docker_compose_up.png?raw=true "docker-compose up")

Containers with `./run start:testing`

![Alt logo](readme_pictures/docker_compose_redis.png?raw=true "docker-compose -f docker-compose-service-with-redis.yml up")

## How to use it

1. Start the service with docker compose

    `./run start`

2. Post xml files to train

    curl -X POST -F 'file=@/PATH/TO/PDF/xml_file_name.xml' localhost:5052/xml_to_train/tenant_name/property_name

3. Post xml files to get suggestions

    curl -X POST -F 'file=@/PATH/TO/PDF/xml_file_name.xml' localhost:5052/xml_to_predict/tenant_name/property_name

![Alt logo](readme_pictures/send_files.png?raw=true "Post xml files")

3. Post labeled data

    curl -X POST --header "Content-Type: application/json" --data '{"xml_file_name": "xml_file_name.xml",
                             "property_name": "property_name",
                             "tenant": "tenant_name",
                             "language_iso": "en",
                             "label_text": "text",
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": [{"left": 124, "top": 48, "width": 83, "height": 13, "page_number": 1}],
                             "label_segments_boxes": [{"left": 124, "top": 48, "width": 83, "height": 13, "page_number": 1}]
                             }' localhost:5052/labeled_data

![Alt logo](readme_pictures/send_json.png?raw=true "Post labeled data")

5. Post data to predict

    curl -X POST --header "Content-Type: application/json" --data '{"xml_file_name": "xml_file_name.xml",
                             "property_name": "property_name",
                             "tenant": "tenant_name",
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": []
                             }' localhost:5052/prediction_data

![Alt logo](readme_pictures/send_json.png?raw=true "Post data to predict")

6. Create model and calculate suggestions

To create the model or calculate the suggestions, a message to redis should be sent. The name for the tasks queue is "
information_extraction_tasks"

    queue = RedisSMQ(host='127.0.0.1', port='6579', qname='information_extraction_tasks', quiet=False)
    # Create model
    queue.sendMessage(delay=0).message('{"tenant": "tenant_name", "task": "create_model", "params": {"property_name": "property_name"}}').execute()
    # Calculate suggestions
    queue.sendMessage(delay=0).message('{"tenant": "tenant_name", "task": "suggestions", "params": {"property_name": "property_name"}}').execute()

![Alt logo](readme_pictures/process.png?raw=true "Create model and calculate suggestions")

7. Get results

There is a redis queue where it is possible to get notified when the different tasks finish

    queue = RedisSMQ(host='127.0.0.1', port='6579', qname='information_extraction_results', quiet=False)
    results_message = queue.receiveMessage().exceptions(False).execute()

    # The models have been created message
    # {"tenant": "tenant_name", 
    # "task": "create_model", 
    # "params": {"property_name": "property_name"}, 
    # "success": true, 
    # "error_message": ""}

    # The suggestions have been computed
    # {"tenant": "tenant_name", 
    # "task": "suggestions", 
    # "params": {"property_name": "property_name"}, 
    # "success": true, 
    # "error_message": "", 
    # "data_url":""}

Get suggestions

    curl -X GET  localhost:5052/get_suggestions/tenant_name/property_name

or in python

    requests.get(results_message.data_url)

![Alt logo](readme_pictures/get_results.png?raw=true "Get results")

The suggestions have the following format:

```
        [{
        "tenant": "tenant", 
        "property_name": "property_name", 
        "xml_file_name": "xml_file_name_1", 
        "text": "suggestion_text_1", 
        "segment_text": "segment_text_1"
        }, 
        {
        "tenant": "tenant", 
        "property_name": "property_name", 
        "xml_file_name": "xml_file_name_2", 
        "text": "suggestion_text_2", 
        "segment_text": "segment_text_2"
        }, ... ]

```

8. Stop the service

    `./run stop`

## HTTP server

![Alt logo](readme_pictures/http_server.png?raw=true "HTTP server")

The container `HTTP server` is coded using Python 3.9 and uses the [FastApi](https://fastapi.tiangolo.com/) web
framework.

If the service is running, the end point definitions can be founded in the following url:

    http://localhost:5052/docs

The end points code can be founded inside the file `app.py`.

The errors are reported to the file `docker_volume/service.log`, if the configuration is not changed (
see [Get service logs](#get-service-logs))

## Queue processor

![Alt logo](readme_pictures/queue_processor.png?raw=true "Queue processor")

The container `Queue processor` is coded using Python 3.9, and it is on charge of the communication with redis.

The code can be founded in the file `QueueProcessor.py` and it uses the library `RedisSMQ` to interact with the redis
queues.

## Service configuration

A configuration file could be provided to set the redis server parameters and the `pdf-information-extraction` server
hosts and ports. If a configuration is not provided, the defaults values uses the redis from the '
docker-compose-service-with-redis.yml' file.

The configuration could be manually created, or it can be used the following script:

    python3 -m pip install graypy~=2.1.0 PyYAML~=5.4.1
    python3 ServiceConfig.py

Configuration file name: `config.yml`

Default parameters:

    service_host: localhost
    service_port: 5052
    redis_host: 127.0.0.1
    redis_port: 6379
    mongo_host: 127.0.0.1
    mongo_port: 29017
    graylog_ip: 

## Get service logs

The service logs are stored by default in the files `docker_volume/service.log` and `docker_volume/redis_tasks.log`

To use a graylog server, add the following line to the `config.yml` file:

    graylog_ip: [ip]

## Set up environment for development

It works with Python 3.9 [install] (https://runnable.com/docker/getting-started/)

    ./run install_venv

## Execute tests

    ./run test

## Troubleshooting

### Issue: Error downloading pip wheel 
Solution: Change RAM memory used by the docker containers to 3Gb or 4Gb 