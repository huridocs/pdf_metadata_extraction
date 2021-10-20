import json
import subprocess
from time import sleep
from unittest import TestCase

from rsmq import RedisSMQ

import requests

from data.InformationExtractionTask import InformationExtractionTask
from data.Params import Params
from data.ResultsMessage import ResultsMessage
from data.Suggestion import Suggestion

DOCKER_VOLUME_PATH = f'./docker_volume'


class TestEndToEnd(TestCase):
    def setUp(self):
        subprocess.run('docker-compose -f docker-compose-service-with-redis.yml up -d --build', shell=True)
        sleep(5)

    def tearDown(self):
        subprocess.run('docker-compose -f docker-compose-service-with-redis.yml down', shell=True)

    def test_end_to_end(self):
        host = 'http://localhost:5052'
        tenant = "end_to_end_test"
        property_name = "property_name"

        with open(f'{DOCKER_VOLUME_PATH}/tenant_test/property_name/xml_to_train/test.xml', 'rb') as stream:
            files = {'file': stream}
            requests.post(f"{host}/xml_to_train/{tenant}/{property_name}", files=files)

        labeled_data_json = {"property_name": property_name,
                             "tenant": tenant,
                             "xml_file_name": "test.xml",
                             "language_iso": "en",
                             "label_text": "text",
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": [],
                             "label_segments_boxes": [{"left": 124, "top": 48, "width": 83, "height": 13,
                                                       "page_number": 1}]
                             }

        requests.post(f"{host}/labeled_data", json=labeled_data_json)

        queue = RedisSMQ(host='127.0.0.1', port='6479', qname='information_extraction_tasks', quiet=False)
        queue.sendMessage().message('{"message_to_avoid":"to_be_written_in_log_file"}').execute()

        task = InformationExtractionTask(tenant=tenant, task='create_model', params=Params(property_name=property_name))
        queue.sendMessage(delay=0).message(str(task.json())).execute()

        results_message = self.get_results_message()
        expected_result = ResultsMessage(tenant=tenant, task='create_model', params=Params(property_name=property_name),
                                         success=True, error_message='', data_url=None)

        self.assertEqual(results_message, expected_result)

        with open(f'{DOCKER_VOLUME_PATH}/tenant_test/property_name/xml_to_predict/test.xml', 'rb') as stream:
            files = {'file': stream}
            requests.post(f"{host}/xml_to_predict/{tenant}/{property_name}", files=files)

        predict_data_json = {"tenant": tenant,
                             "property_name": property_name,
                             "xml_file_name": "test.xml",
                             "page_width": 612,
                             "page_height": 792,
                             "xml_segments_boxes": [],
                             }

        requests.post(f"{host}/prediction_data", json=predict_data_json)

        task = InformationExtractionTask(tenant=tenant, task='suggestions', params=Params(property_name=property_name))
        queue.sendMessage(delay=0).message(str(task.json())).execute()

        results_message = self.get_results_message()
        expected_result = ResultsMessage(tenant=tenant,
                                         task='suggestions',
                                         params=Params(property_name=property_name),
                                         success=True,
                                         error_message='',
                                         data_url=f"{host}/get_suggestions/{tenant}/{property_name}")

        self.assertEqual(results_message, expected_result)

        response = requests.get(results_message.data_url)

        suggestions = json.loads(response.json())
        suggestion = Suggestion(**suggestions[0])

        self.assertEqual(1, len(suggestions))

        self.assertEqual(tenant, suggestion.tenant)
        self.assertEqual(property_name, suggestion.property_name)
        self.assertEqual('test.xml', suggestion.xml_file_name)
        self.assertEqual('United Nations', suggestion.text)
        self.assertEqual('United Nations', suggestion.segment_text)

        task = InformationExtractionTask(tenant=tenant, task='create_model', params=Params(property_name=property_name))
        queue.sendMessage(delay=0).message(str(task.json())).execute()

        results_message = self.get_results_message()
        expected_result = ResultsMessage(tenant=tenant, task='create_model', params=Params(property_name=property_name),
                                         success=False, error_message='No labeled data to create model', data_url=None)

        self.assertEqual(results_message, expected_result)

        task = InformationExtractionTask(tenant=tenant, task='suggestions', params=Params(property_name=property_name))
        queue.sendMessage(delay=0).message(str(task.json())).execute()

        results_message = self.get_results_message()
        expected_result = ResultsMessage(tenant=tenant, task='suggestions', params=Params(property_name=property_name),
                                         success=False, error_message='No data to calculate suggestions', data_url=None)

        self.assertEqual(results_message, expected_result)

    @staticmethod
    def get_results_message() -> ResultsMessage:
        for i in range(10):
            sleep(1)
            queue = RedisSMQ(host='127.0.0.1', port='6479', qname='information_extraction_results', quiet=False)
            message = queue.receiveMessage().exceptions(False).execute()
            if message:
                queue.deleteMessage(id=message['id']).execute()
                return ResultsMessage(**json.loads(message['message']))