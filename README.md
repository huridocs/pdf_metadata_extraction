# PDF information extraction

Project to extract information extraction 

### Execute tests

    `python -m unittest`

### How to use it

Start service:

  `docker-compose up`

Post xml file:

   `curl -X POST -F 'file=@/PATH/TO/PDF/xml_file_name.xml' localhost:5050/xml_file/tenant_name/property_name`

Post labeled data:

   `curl -X POST --header "Content-Type: application/json" --data '{"xml_file_name": "xml_file_name.xml",
                             "property_name": "property_name",
                             "tenant": "tenant_name",
                             "language_iso": "en",
                             "label_text": "text",
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": [],
                             "label_segments_boxes": [{"left": 124, "top": 48, "width": 83, "height": 13, "page_number": 1}]
                             }' localhost:5050/labeled_data`

Post prediction data:

   `curl -X POST --header "Content-Type: application/json" --data '{"xml_file_name": "xml_file_name.xml",
                             "property_name": "property_name",
                             "tenant": "tenant_name",
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": []
                             }' localhost:5050/prediction_data`

Calculate suggestions:

`curl -X POST  localhost:5050/calculate_suggestions/tenant_name/property_name`

Get suggestions:

`curl -X GET  localhost:5050/get_suggestions/tenant_name/property_name`


To stop the server:

  `docker-compose down`
  

### Configuring external graylog server 

Set graylog ip in the ./graylog.yml file:

   `graylog_ip: [ip]`