import json
import subprocess
from time import sleep
from unittest import TestCase

import requests

from data.Suggestion import Suggestion

DOCKER_VOLUME_PATH = f'./docker_volume'


class TestEndToEnd(TestCase):
    def test_end_to_end(self):
        tenant = "end_to_end_test"
        extraction_name = "property_name"
        
        host = 'http://localhost:5050'

        subprocess.run('docker-compose up -d', shell=True)
        sleep(20)

        with open(f'{DOCKER_VOLUME_PATH}/tenant_test/extraction_name/xml_files/test.xml', 'rb') as stream:
            files = {'file': stream}
            requests.post(f"{host}/xml_file/{tenant}/{extraction_name}", files=files)

        predict_data_json = {"xml_file_name": "test.xml",
                             "extraction_name": extraction_name,
                             "tenant": tenant,
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": [],
                             }

        labeled_data_json = {"xml_file_name": "test.xml",
                             "extraction_name": extraction_name,
                             "tenant": tenant,
                             "language_iso": "en",
                             "label_text": "text",
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": [],
                             "label_segments_boxes": [{"left": 124, "top": 48, "width": 83, "height": 13,
                                                       "page_number": 1}]
                             }

        requests.post(f"{host}/prediction_data", json=predict_data_json)
        requests.post(f"{host}/labeled_data", json=labeled_data_json)
        requests.post(f"{host}/add_task/", json={
            "extraction_name": extraction_name,
            "tenant": tenant,
        })

        sleep(10)

        response = requests.get(f"{host}/get_suggestions/{tenant}/{extraction_name}")
        suggestions_dict = json.loads(response.json())

        suggestions = [Suggestion(**suggestions_dict[0]), Suggestion(**suggestions_dict[1])]

        self.assertEqual(2, len(suggestions))

        self.assertEqual({tenant}, {x.tenant for x in suggestions})
        self.assertEqual({extraction_name}, {x.extraction_name for x in suggestions})
        self.assertEqual({"test.xml"}, {x.xml_file_name for x in suggestions})
        self.assertEqual({"United Nations"}, {x.segment_text for x in suggestions})
        self.assertEqual({"United Nations"}, {x.text for x in suggestions})

        subprocess.run('docker-compose down', shell=True)
