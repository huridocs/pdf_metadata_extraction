from fastapi.testclient import TestClient
from unittest import TestCase
from app import app

client = TestClient(app)


class TestApp(TestCase):
    def test_info(self):
        response = client.get("/info")
        self.assertEqual(200, response.status_code)

    def test_error(self):
        response = client.get("/error")
        self.assertEqual(500, response.status_code)
        self.assertEqual({'detail': 'This is a test error from the error endpoint'}, response.json())