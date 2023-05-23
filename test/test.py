from main import app
import  unittest
from fastapi.testclient import TestClient


class FastApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client = None

    def test_read_main(self):
        response = self.client.get("/items/items/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_read_one(self):
        response = self.client.get("/items/items/?limit=1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)


if __name__ == "__main__":
    unittest.main()