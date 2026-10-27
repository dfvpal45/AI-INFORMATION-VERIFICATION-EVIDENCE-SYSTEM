import re
import unittest

from fastapi.testclient import TestClient

from app.main import app, frontend_dist


class SingleServiceDeploymentTests(unittest.TestCase):
    @unittest.skipUnless(frontend_dist.is_dir(), 'Build the frontend to test static serving.')
    def test_root_serves_frontend_and_built_assets(self):
        client = TestClient(app)
        response = client.get('/')

        self.assertEqual(response.status_code, 200)
        self.assertIn('<div id="root"></div>', response.text)

        asset = re.search(r'(?:src|href)="(/assets/[^"]+)"', response.text)
        self.assertIsNotNone(asset)
        asset_response = client.get(asset.group(1))
        self.assertEqual(asset_response.status_code, 200)

    def test_health_endpoint_remains_available(self):
        response = TestClient(app).get('/api/health')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
