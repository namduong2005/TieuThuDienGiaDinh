import unittest

import numpy as np
import pandas as pd

from src.features import DailyShapeNormalizer, build_preprocessor
from src.data import season_from_month
from src.train import (
    choose_configuration,
    chronological_split,
    evaluate_experiments,
)


class FeatureTests(unittest.TestCase):
    def test_shape_normalizer_makes_row_means_one(self):
        values = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
        transformed = DailyShapeNormalizer().fit_transform(values)
        np.testing.assert_allclose(transformed.mean(axis=1), np.ones(2))

    def test_preprocessor_imputes_missing_values(self):
        values = np.array([[1.0, np.nan, 3.0], [2.0, 4.0, 6.0]])
        transformed = build_preprocessor("absolute").fit_transform(values)
        self.assertFalse(np.isnan(transformed).any())

    def test_imputer_is_fitted_only_on_training_values(self):
        train = pd.DataFrame([[1.0, np.nan], [3.0, 5.0]])
        validation = pd.DataFrame([[100.0, np.nan]])
        preprocessor = build_preprocessor("absolute")
        preprocessor.fit(train)
        preprocessor.transform(validation)
        statistics = preprocessor.named_steps["imputer"].statistics_
        np.testing.assert_allclose(statistics, [2.0, 5.0])


class SplitTests(unittest.TestCase):
    def test_split_is_chronological_and_disjoint(self):
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100, freq="D"),
                "value": range(100),
            }
        )
        train, validation, test = chronological_split(frame)
        self.assertLess(train["date"].max(), validation["date"].min())
        self.assertLess(validation["date"].max(), test["date"].min())
        self.assertEqual(len(train) + len(validation) + len(test), len(frame))


class ExperimentTests(unittest.TestCase):
    @staticmethod
    def make_profiles(number_of_days=60):
        rng = np.random.default_rng(7)
        hours = np.arange(24)
        rows = []
        for index in range(number_of_days):
            center = 7 if index % 2 == 0 else 19
            values = 0.4 + np.exp(-0.5 * ((hours - center) / 2.5) ** 2)
            values += rng.normal(0, 0.03, size=24)
            row = {"date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=index)}
            row.update({f"hour_{hour:02d}": values[hour] for hour in hours})
            rows.append(row)
        return pd.DataFrame(rows)

    def test_experiment_grid_covers_modes_and_k_values(self):
        frame = self.make_profiles()
        train, validation, _ = chronological_split(frame)
        experiments = evaluate_experiments(train, validation, [2, 3], [0, 1])
        self.assertEqual(len(experiments), 4)
        self.assertEqual(set(experiments["mode"]), {"absolute", "shape"})
        self.assertEqual(set(experiments["k"]), {2, 3})
        self.assertTrue(experiments["validation_silhouette_mean"].notna().all())
        self.assertIn("validation_silhouette_std", experiments.columns)
        self.assertIn("stability_ari_std", experiments.columns)
        self.assertTrue((experiments["validation_silhouette_std"] >= 0).all())

    def test_configuration_rejects_tiny_clusters_when_possible(self):
        experiments = pd.DataFrame(
            [
                {
                    "mode": "absolute",
                    "k": 8,
                    "validation_silhouette_mean": 0.9,
                    "minimum_cluster_fraction_mean": 0.01,
                    "stability_ari_mean": 1.0,
                    "validation_distance_mean": 0.2,
                },
                {
                    "mode": "absolute",
                    "k": 2,
                    "validation_silhouette_mean": 0.7,
                    "minimum_cluster_fraction_mean": 0.25,
                    "stability_ari_mean": 1.0,
                    "validation_distance_mean": 0.3,
                },
            ]
        )
        selected = choose_configuration(experiments)
        self.assertEqual(int(selected["k"]), 2)

    def test_season_mapping(self):
        self.assertEqual(season_from_month(1), "winter")
        self.assertEqual(season_from_month(4), "spring")
        self.assertEqual(season_from_month(7), "summer")
        self.assertEqual(season_from_month(10), "autumn")


if __name__ == "__main__":
    unittest.main()
