<h3 align="center">PDF metadata extraction</h3>
<p align="center">A Docker-powered service for extracting pieces of information from PDFs</p>

---

## Contents

- [Dependencies](#dependencies)
- [Requirements](#requirements)
- [Docker containers](#docker-containers)
- [How to use it](#how-to-use-it)
- [How to use GPU](#how-to-use-gpu)
- [HTTP server](#http-server)
- [Queue processor](#queue-processor)
- [Service configuration](#service-configuration)
- [Get service logs](#get-service-logs)
- [Set up environment for development](#set-up-environment-for-development)
- [Execute tests](#execute-tests)
- [Execute performance test](#execute-performance-test)
- [Troubleshooting](#troubleshooting)

## Dependencies
* Docker 23.0.1 [install link](https://runnable.com/docker/getting-started/)
* Docker-compose 

    Note: On mac Docker-compose is installed with Docker

    [install on Ubuntu link](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-compose-on-ubuntu-20-04)

## Requirements

* 12Gb RAM memory

## Docker containers

A redis server is needed to use the service asynchronously. For that matter, it can be used the 
command `make start:testing` that has a built-in 
redis server.

Containers with `make start`

![Alt logo](readme_pictures/docker_compose_up.png?raw=true "docker-compose up")

Containers with `make start_for_testing`

![Alt logo](readme_pictures/docker_compose_redis.png?raw=true "docker-compose -f docker-compose-service-with-redis.yml up")

## How to use it

1. Start the service with docker compose

    `make start`

2. Post xml files to train

    curl -X POST -F 'file=@/PATH/TO/PDF/xml_file_name.xml' localhost:5056/xml_to_train/tenant_name/property_name

3. Post xml files to get suggestions

    curl -X POST -F 'file=@/PATH/TO/PDF/xml_file_name.xml' localhost:5056/xml_to_predict/tenant_name/property_name

![Alt logo](readme_pictures/send_files.png?raw=true "Post xml files")

3. Post labeled data
    
    Text, numeric or date cases:

    curl -X POST --header "Content-Type: application/json" --data '{"xml_file_name": "xml_file_name.xml",
                             "property_name": "property_name",
                             "tenant": "tenant_name",
                             "language_iso": "en",
                             "label_text": "text",
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": [{"left": 124, "top": 48, "width": 83, "height": 13, "page_number": 1}],
                             "label_segments_boxes": [{"left": 124, "top": 48, "width": 83, "height": 13, "page_number": 1}]
                             }' localhost:5056/labeled_data
    
    Multi-option case:

    curl -X POST --header "Content-Type: application/json" --data '{"xml_file_name": "xml_file_name.xml",
                             "property_name": "property_name",
                             "tenant": "tenant_name",
                             "language_iso": "en",
                             "options": [{"id": "1", "label": "option 1"}, {"id": "2", "label": "option 2"}],
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": [{"left": 124, "top": 48, "width": 83, "height": 13, "page_number": 1}],
                             "label_segments_boxes": [{"left": 124, "top": 48, "width": 83, "height": 13, "page_number": 1}]
                             }' localhost:5056/labeled_data

![Alt logo](readme_pictures/send_json.png?raw=true "Post labeled data")

4. Post data to predict

    curl -X POST --header "Content-Type: application/json" --data '{"xml_file_name": "xml_file_name.xml",
                             "property_name": "property_name",
                             "tenant": "tenant_name",
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": []
                             }' localhost:5056/prediction_data

![Alt logo](readme_pictures/send_json.png?raw=true "Post data to predict")

5. Create model and calculate suggestions

To create the model or calculate the suggestions, a message to redis should be sent. The name for the tasks queue is "
information_extraction_tasks"

    queue = RedisSMQ(host='127.0.0.1', port='6579', qname='information_extraction_tasks', quiet=False)
    
    # Text, numeric or date cases:

    # Create model
    queue.sendMessage(delay=0).message('{"tenant": "tenant_name", "task": "create_model", "params": {"property_name": "property_name"}}').execute()
    # Calculate suggestions
    queue.sendMessage(delay=0).message('{"tenant": "tenant_name", "task": "suggestions", "params": {"property_name": "property_name"}}').execute()
    
    # Multi-option case:

    # Create model
    # The options parameter are all the posible values for all the PDF
    # The multi_value parameter tells if the algorithm can pick more than one option per PDF
    queue.sendMessage(delay=0).message('{"tenant": "tenant_name", "task": "create_model", "params": {"property_name": "property_name" , "options": [{"id": "1", "label": "option 1"}, {"id": "2", "label": "option 2"}, {"id": "3", "label": "option 3"}], "multi_value": false}}').execute()
    # Calculate suggestions
    queue.sendMessage(delay=0).message('{"tenant": "tenant_name", "task": "suggestions", "params": {"property_name": "property_name"}}').execute()

![Alt logo](readme_pictures/process.png?raw=true "Create model and calculate suggestions")

6. Get results

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

    curl -X GET  localhost:5056/get_suggestions/tenant_name/property_name

or in python

    requests.get(results_message.data_url)

![Alt logo](readme_pictures/get_results.png?raw=true "Get results")

The suggestions have the following format:

Text, numeric or date cases:

```
        [{
        "tenant": "tenant", 
        "property_name": "property_name", 
        "xml_file_name": "xml_file_name_1", 
        "text": "suggestion_text_1", 
        "segment_text": "segment_text_1",
        "segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 1}]
        }, 
        {
        "tenant": "tenant", 
        "property_name": "property_name", 
        "xml_file_name": "xml_file_name_2", 
        "text": "suggestion_text_2", 
        "segment_text": "segment_text_2",
        "segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 2}]
        }, ... ]

```

Multi-option case:

```
        [{
        "tenant": "tenant", 
        "property_name": "property_name", 
        "xml_file_name": "xml_file_name_1", 
        "options": [{"id": "1", "label": "option 1"}], 
        "segment_text": "segment_text_1",
        "segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 1}]
        }, 
        {
        "tenant": "tenant", 
        "property_name": "property_name", 
        "xml_file_name": "xml_file_name_2", 
        "options": [{"id": "2", "label": "option 2"}], 
        "segment_text": "segment_text_2",
        "segments_boxes": [{"left": 1, "top": 2, "width": 3, "height": 4, "page_number": 2}]
        }, ... ]

```

8. Stop the service

    `make stop`


## How to use GPU

To use the GPU in the docker containers

1. Install the package:
        
    nvidia-container-toolkit


2. Restart docker service

    systemctl restart docker



3. Start the service with
    
    make start_gpu

## HTTP server

![Alt logo](readme_pictures/http_server.png?raw=true "HTTP server")

The container `HTTP server` is coded using Python 3.9 and uses the [FastApi](https://fastapi.tiangolo.com/) web
framework.

If the service is running, the end point definitions can be founded in the following url:

    http://localhost:5056/docs

The end points code can be founded inside the file `app.py`.

The errors are reported to the file `docker_volume/service.log`, if the configuration is not changed (
see [Get service logs](#get-service-logs))

## Queue processor

![Alt logo](readme_pictures/queue_processor.png?raw=true "Queue processor")

The container `Queue processor` is coded using Python 3.9, and it is on charge of the communication with redis.

The code can be founded in the file `QueueProcessor.py` and it uses the library `RedisSMQ` to interact with the redis
queues.

## Service configuration

See environment variables in the file .env

## Set up environment for development

It works with Python 3.9 [install] (https://runnable.com/docker/getting-started/)

    make install_venv

## Execute tests

    make start_for_testing
    make test

## Execute performance test

    cd src && python check_performance.py

And the results are stored in src/performance/results

## Troubleshooting

### Issue: Error downloading pip wheel 
Solution: Change RAM memory used by the docker containers to 3Gb or 4Gb 