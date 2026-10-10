import unittest

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from src.config import HOUR_COLUMNS
from src.features import build_preprocessor
from src.train import choose_configuration, chronological_split, evaluate_experiments


class EndToEndTests(unittest.TestCase):
    def test_training_pipeline_from_daily_profiles_to_test_predictions(self):
        rng = np.random.default_rng(21)
        hours = np.arange(24)
        rows = []
        for index, date in enumerate(pd.date_range("2021-01-01", periods=90)):
            center = (7, 14, 20)[index % 3]
            values = 0.5 + 1.3 * np.exp(-0.5 * ((hours - center) / 2.8) ** 2)
            values += rng.normal(0, 0.04, 24)
            row = {"date": date, "is_weekend": int(date.dayofweek >= 5)}
            row.update({column: values[position] for position, column in enumerate(HOUR_COLUMNS)})
            rows.append(row)

        frame = pd.DataFrame(rows)
        train, validation, test = chronological_split(frame)
        experiments = evaluate_experiments(train, validation, [2, 3], [0, 1])
        selected = choose_configuration(experiments)
        combined = pd.concat([train, validation], ignore_index=True)
        pipeline = Pipeline(
            [
                ("preprocess", build_preprocessor(str(selected["mode"]))),
                ("kmeans", KMeans(n_clusters=int(selected["k"]), n_init=10, random_state=42)),
            ]
        )
        pipeline.fit(combined[HOUR_COLUMNS])
        predictions = pipeline.predict(test[HOUR_COLUMNS])

        self.assertEqual(len(predictions), len(test))
        self.assertFalse(np.isnan(predictions).any())
        self.assertGreaterEqual(len(np.unique(predictions)), 2)


if __name__ == "__main__":
    unittest.main()
