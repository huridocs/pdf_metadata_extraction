import json
import subprocess
from time import sleep
from unittest import TestCase
from rsmq import RedisSMQ

import requests

from data.Suggestion import Suggestion
from data.SuggestionsCalculatedMessage import SuggestionsCalculatedMessage

DOCKER_VOLUME_PATH = f'./docker_volume'


class TestEndToEnd(TestCase):
    def test_end_to_end(self):
        tenant = "end_to_end_test"
        template = "template_test"
        property_name = "property_name"

        host = 'http://localhost:5050'

        subprocess.run('docker-compose -f docker-compose-redis.yml up -d', shell=True)
        sleep(5)

        labeled_data_json = {"xml_file_name": "test.xml",
                             "property_name": property_name,
                             "tenant": tenant,
                             "language_iso": "en",
                             "label_text": "text",
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": [],
                             "label_segments_boxes": [{"left": 124, "top": 48, "width": 83, "height": 13,
                                                       "page_number": 1}]
                             }

        with open(f'{DOCKER_VOLUME_PATH}/tenant_test/property_name/xml_files/test.xml', 'rb') as stream:
            files = {'file': stream}
            requests.post(f"{host}/training_xml_file/{tenant}/{property_name}", files=files)

        requests.post(f"{host}/labeled_data", json=labeled_data_json)
        requests.post(f"{host}/create_model/", json={
            "property_name": property_name,
            "tenant": tenant,
        })

        predict_data_json = {"tenant": tenant,
                             "template": template,
                             "property_name": property_name,
                             "xml_file_name": "test.xml",
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": [],
                             }

        with open(f'{DOCKER_VOLUME_PATH}/tenant_test/property_name/xml_files/test.xml', 'rb') as stream:
            files = {'file': stream}
            requests.post(f"{host}/predict_xml_file/{tenant}/{property_name}", files=files)

        requests.post(f"{host}/prediction_data", json=predict_data_json)

        suggestion_calculated_message = self.get_redis_message()

        response = requests.get(
            f"{host}/get_suggestions/{suggestion_calculated_message.tenant}/{suggestion_calculated_message.template}/{suggestion_calculated_message.property_name}")

        suggestions = json.loads(response.json())
        suggestion = Suggestion(**suggestions[0])

        self.assertEqual(1, len(suggestions))

        self.assertEqual(tenant, suggestion.tenant)
        self.assertEqual(template, suggestion.template)
        self.assertEqual(property_name, suggestion.property_name)
        self.assertEqual('test.xml', suggestion.xml_file_name)
        self.assertEqual('United Nations', suggestion.text)
        self.assertEqual('United Nations', suggestion.segment_text)

        subprocess.run('docker-compose -f docker-compose-redis.yml down', shell=True)

    @staticmethod
    def get_redis_message() -> SuggestionsCalculatedMessage:
        for i in range(10):
            sleep(1)
            queue = RedisSMQ(host='127.0.0.1', port='6479', qname="suggestions_calculated", quiet=True)
            message = queue.receiveMessage().exceptions(False).execute()
            if message:
                return SuggestionsCalculatedMessage(**json.loads(message['message']))
