# PDF information extraction

Project to extract information extraction 

### Execute tests

    python -m unittest

### How to use it

<b>Configure the redis server</b>

If the configuration is not changed, a dockerized redis server will be used.

To use a different redis server, create a file `docker_volume/redis_server.yml` with the following content:

    host: [shost_ip]
    port: [port_number]

<b>Start service</b>

    docker-compose up

<b>Post xml file</b>

    curl -X POST -F 'file=@/PATH/TO/PDF/xml_file_name.xml' localhost:5050/xml_file/tenant_name/property_name

<b>Post labeled data</b>

    curl -X POST --header "Content-Type: application/json" --data '{"xml_file_name": "xml_file_name.xml",
                             "property_name": "property_name",
                             "tenant": "tenant_name",
                             "language_iso": "en",
                             "label_text": "text",
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": [{"left": 124, "top": 48, "width": 83, "height": 13, "page_number": 1}],
                             "label_segments_boxes": [{"left": 124, "top": 48, "width": 83, "height": 13, "page_number": 1}]
                             }' localhost:5050/labeled_data

<b>Post prediction data</b>

    curl -X POST --header "Content-Type: application/json" --data '{"xml_file_name": "xml_file_name.xml",
                             "property_name": "property_name",
                             "tenant": "tenant_name",
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": []
                             }' localhost:5050/prediction_data

<b>Create model and calculate suggestions</b>

To create the model or calculate the suggestions, a message to redis should be sent. The name for the tasks queue is "information_extraction_tasks"

    queue = RedisSMQ(host='127.0.0.1', port='6479', qname='information_extraction_tasks', quiet=False)
    # Create model
    queue.sendMessage(delay=0).message('{"tenant": "tenant_name", "task": "create_model", "data": {"property_name": "property_name"}}').execute()
    # Calculate suggestions
    queue.sendMessage(delay=0).message('{"tenant": "tenant_name", "task": "suggestions", "data": {"property_name": "property_name"}}').execute()

<b>Notification</b>

There is a redis queue where it is possible to get notified when the different tasks finish

    queue = RedisSMQ(host='127.0.0.1', port='6479', qname='information_extraction_results', quiet=False)
    message = queue.receiveMessage().exceptions(False).execute()

    # The models have been created message
    # {"tenant": "tenant_name", "task": "create_model", "data": {"property_name": "property_name"}, "success": true, "error_message": ""}

    # The suggestions have been computed
    # {"tenant": "tenant_name", "task": "suggestions", "data": {"property_name": "property_name"}, "success": true, "error_message": ""}

<b>Get suggestions</b>

    curl -X GET  localhost:5050/get_suggestions/tenant_name/property_name


<b>To stop the server</b>

    docker-compose down
  

### Logging

The service logs are stored in the file `docker_volume/service.log`

To use a graylog server, create a file `docker_volume/graylog.yml` with the following content:

`graylog_ip: [ip]`