import unittest

import numpy as np
import pandas as pd

from src.features import DailyShapeNormalizer, build_preprocessor
from src.train import chronological_split


class FeatureTests(unittest.TestCase):
    def test_shape_normalizer_makes_row_means_one(self):
        values = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
        transformed = DailyShapeNormalizer().fit_transform(values)
        np.testing.assert_allclose(transformed.mean(axis=1), np.ones(2))

    def test_preprocessor_imputes_missing_values(self):
        values = np.array([[1.0, np.nan, 3.0], [2.0, 4.0, 6.0]])
        transformed = build_preprocessor("absolute").fit_transform(values)
        self.assertFalse(np.isnan(transformed).any())


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


if __name__ == "__main__":
    unittest.main()
