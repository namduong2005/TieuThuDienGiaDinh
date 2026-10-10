import unittest

from fastapi.testclient import TestClient

from src.api import app
from src.config import MODEL_PATH


@unittest.skipUnless(MODEL_PATH.exists(), "Chưa có mô hình thật để kiểm thử API")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.valid_profile = [
            0.45, 0.42, 0.39, 0.38, 0.37, 0.40,
            0.55, 0.82, 0.76, 0.63, 0.58, 0.56,
            0.54, 0.53, 0.55, 0.61, 0.72, 0.95,
            1.35, 1.72, 1.84, 1.50, 1.02, 0.68,
        ]

    def test_health_reports_ready_model(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")

    def test_home_page_serves_the_web_interface(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Phân tích hồ sơ điện một ngày", response.text)

    def test_all_web_pages_serve_the_interface(self):
        for path in ("/analyze", "/calendar", "/dashboard", "/data-quality"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/html", response.headers["content-type"])

    def test_day_profile_and_calendar_data(self):
        response = self.client.get("/api/day-profile?date=2010-11-26")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["hourly_power"]), 24)
        self.assertEqual(self.client.get("/api/day-profile?date=1999-01-01").status_code, 404)
        winter = self.client.get("/api/days?season=winter")
        self.assertEqual(winter.status_code, 200)
        self.assertTrue(all(day["season"] == "winter" for day in winter.json()["days"]))

    def test_dashboard_and_quality_endpoints(self):
        dashboard = self.client.get("/api/dashboard")
        self.assertEqual(dashboard.status_code, 200)
        self.assertTrue(dashboard.json()["experiments"])
        self.assertTrue(dashboard.json()["cluster_sizes"])
        quality = self.client.get("/api/data-quality")
        self.assertEqual(quality.status_code, 200)
        self.assertIn("removed_days", quality.json())
        self.assertEqual(quality.json()["eda"]["eda_scope"], "train_only")

    def test_predict_returns_cluster_and_distance(self):
        response = self.client.post(
            "/predict", json={"hourly_power": self.valid_profile}
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body["cluster"], [0, 1])
        self.assertGreaterEqual(body["distance_to_centroid"], 0)
        self.assertEqual(len(body["centroid_profile"]), 24)

    def test_predict_accepts_up_to_two_missing_hours(self):
        profile = self.valid_profile.copy()
        profile[3] = None
        profile[18] = None
        response = self.client.post("/predict", json={"hourly_power": profile})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["imputed_hours"], [3, 18])

    def test_predict_rejects_wrong_number_of_hours(self):
        response = self.client.post(
            "/predict", json={"hourly_power": self.valid_profile[:23]}
        )
        self.assertEqual(response.status_code, 422)

    def test_predict_rejects_negative_power(self):
        profile = self.valid_profile.copy()
        profile[0] = -1
        response = self.client.post("/predict", json={"hourly_power": profile})
        self.assertEqual(response.status_code, 422)

    def test_predict_warns_for_values_outside_training_domain(self):
        response = self.client.post("/predict", json={"hourly_power": [100.0] * 24})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["outside_training_domain"])
        self.assertTrue(response.json()["warnings"])


if __name__ == "__main__":
    unittest.main()
