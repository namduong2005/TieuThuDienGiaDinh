import argparse
import json
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline

from src.config import (
    ARTIFACT_VARIANT,
    ASSIGNMENTS_PATH,
    CENTROIDS_PATH,
    DAILY_PROFILES_PATH,
    EXPERIMENTS_PATH,
    HOUR_COLUMNS,
    MODEL_PATH,
    QUALITY_PATH,
    SUB_METERING_COLUMNS,
    SUMMARY_PATH,
    ensure_directories,
)
from src.features import build_preprocessor


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
):
    ordered = frame.sort_values("date").reset_index(drop=True)
    number_of_rows = len(ordered)
    if number_of_rows < 30:
        raise ValueError("Can it nhat 30 ngay de chia train/validation/test")

    train_end = max(1, int(number_of_rows * train_fraction))
    validation_end = max(train_end + 1, int(number_of_rows * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, number_of_rows - 1)

    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )


def safe_silhouette(X, labels) -> float:
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return float("nan")
    if counts.min() < 2:
        return float("nan")
    return float(silhouette_score(X, labels))


def cluster_stability(label_runs) -> float:
    if len(label_runs) < 2:
        return 1.0
    scores = [
        adjusted_rand_score(first, second)
        for first, second in combinations(label_runs, 2)
    ]
    return float(np.mean(scores))


def cluster_stability_statistics(label_runs) -> dict[str, float]:
    if len(label_runs) < 2:
        scores = np.array([1.0])
    else:
        scores = np.asarray([
            adjusted_rand_score(first, second)
            for first, second in combinations(label_runs, 2)
        ], dtype=float)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=0)),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }


def mean_nearest_centroid_distance(model: KMeans, X) -> float:
    return float(model.transform(X).min(axis=1).mean())


def nanmean_or_nan(values) -> float:
    array = np.asarray(values, dtype=float)
    valid = array[~np.isnan(array)]
    return float(valid.mean()) if len(valid) else float("nan")


def nan_statistics(values) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    valid = array[~np.isnan(array)]
    if not len(valid):
        return {key: float("nan") for key in ("mean", "std", "min", "max")}
    return {
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=0)),
        "min": float(valid.min()),
        "max": float(valid.max()),
    }


def evaluate_experiments(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    k_values,
    seeds,
) -> pd.DataFrame:
    X_train_raw = train_frame[HOUR_COLUMNS]
    X_validation_raw = validation_frame[HOUR_COLUMNS]
    records = []

    for mode in ("absolute", "shape"):
        preprocessor = build_preprocessor(mode)
        X_train = preprocessor.fit_transform(X_train_raw)
        X_validation = preprocessor.transform(X_validation_raw)

        for k in k_values:
            train_silhouettes = []
            validation_silhouettes = []
            train_inertias = []
            validation_distances = []
            label_runs = []
            minimum_cluster_fractions = []

            for seed in seeds:
                model = KMeans(n_clusters=k, random_state=seed, n_init=20)
                train_labels = model.fit_predict(X_train)
                validation_labels = model.predict(X_validation)
                label_runs.append(train_labels)

                counts = np.bincount(train_labels, minlength=k)
                minimum_cluster_fractions.append(float(counts.min() / len(train_labels)))
                train_silhouettes.append(safe_silhouette(X_train, train_labels))
                validation_silhouettes.append(
                    safe_silhouette(X_validation, validation_labels)
                )
                train_inertias.append(float(model.inertia_ / len(X_train)))
                validation_distances.append(
                    mean_nearest_centroid_distance(model, X_validation)
                )

            train_stats = nan_statistics(train_silhouettes)
            validation_stats = nan_statistics(validation_silhouettes)
            inertia_stats = nan_statistics(train_inertias)
            distance_stats = nan_statistics(validation_distances)
            stability_stats = cluster_stability_statistics(label_runs)
            records.append(
                {
                    "mode": mode,
                    "k": k,
                    **{f"train_silhouette_{key}": value for key, value in train_stats.items()},
                    **{f"validation_silhouette_{key}": value for key, value in validation_stats.items()},
                    **{f"train_inertia_per_day_{key}": value for key, value in inertia_stats.items()},
                    **{f"validation_distance_{key}": value for key, value in distance_stats.items()},
                    **{f"stability_ari_{key}": value for key, value in stability_stats.items()},
                    "minimum_cluster_fraction_mean": np.mean(minimum_cluster_fractions),
                    "number_of_seeds": len(seeds),
                }
            )

    return pd.DataFrame(records)


def choose_configuration(experiments: pd.DataFrame) -> pd.Series:
    eligible = experiments[
        experiments["validation_silhouette_mean"].notna()
        & (experiments["minimum_cluster_fraction_mean"] >= 0.03)
    ].copy()
    if eligible.empty:
        eligible = experiments[experiments["validation_silhouette_mean"].notna()].copy()
    if eligible.empty:
        raise RuntimeError("Khong co cau hinh K-means hop le")

    eligible = eligible.sort_values(
        [
            "validation_silhouette_mean",
            "stability_ari_mean",
            "validation_distance_mean",
        ],
        ascending=[False, False, True],
    )
    return eligible.iloc[0]


def split_label(date, train_end, validation_end) -> str:
    if date <= train_end:
        return "train"
    if date <= validation_end:
        return "validation"
    return "test"


def main():
    parser = argparse.ArgumentParser(description="Chon K va huan luyen K-means")
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--final-seed", type=int, default=42)
    args = parser.parse_args()

    ensure_directories()
    if not DAILY_PROFILES_PATH.exists():
        raise FileNotFoundError(
            "Chua co daily_profiles.csv. Chay: py -m src.data --demo "
            "hoac py -m src.data --download --prepare"
        )

    frame = pd.read_csv(DAILY_PROFILES_PATH, parse_dates=["date"])
    missing_columns = [column for column in HOUR_COLUMNS if column not in frame]
    if missing_columns:
        raise ValueError(f"Thieu cac cot gio: {missing_columns}")

    train_frame, validation_frame, test_frame = chronological_split(frame)
    k_values = range(args.min_k, args.max_k + 1)
    seeds = range(args.seeds)

    experiments = evaluate_experiments(
        train_frame,
        validation_frame,
        k_values,
        seeds,
    )
    experiments.to_csv(EXPERIMENTS_PATH, index=False)
    selected = choose_configuration(experiments)
    selected_mode = str(selected["mode"])
    selected_k = int(selected["k"])

    train_validation = pd.concat([train_frame, validation_frame], ignore_index=True)
    final_pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(selected_mode)),
            (
                "kmeans",
                KMeans(
                    n_clusters=selected_k,
                    random_state=args.final_seed,
                    n_init=20,
                ),
            ),
        ]
    )
    final_pipeline.fit(train_validation[HOUR_COLUMNS])
    joblib.dump(final_pipeline, MODEL_PATH)

    X_test_transformed = final_pipeline.named_steps["preprocess"].transform(
        test_frame[HOUR_COLUMNS]
    )
    test_labels = final_pipeline.named_steps["kmeans"].predict(X_test_transformed)
    test_silhouette = safe_silhouette(X_test_transformed, test_labels)
    test_distance = mean_nearest_centroid_distance(
        final_pipeline.named_steps["kmeans"], X_test_transformed
    )

    all_assignments = frame.copy()
    all_assignments["cluster"] = final_pipeline.predict(frame[HOUR_COLUMNS])
    train_end = train_frame["date"].max()
    validation_end = validation_frame["date"].max()
    all_assignments["split"] = all_assignments["date"].map(
        lambda date: split_label(date, train_end, validation_end)
    )
    assignment_columns = [
        "date",
        "split",
        "cluster",
        "missing_hours",
        "day_of_week",
        "is_weekend",
        "month",
        "season",
    ]
    assignment_columns.extend(
        column for column in SUB_METERING_COLUMNS if column in all_assignments
    )
    if "demo_true_pattern" in all_assignments:
        assignment_columns.append("demo_true_pattern")
    all_assignments[assignment_columns].to_csv(ASSIGNMENTS_PATH, index=False)

    centers_scaled = final_pipeline.named_steps["kmeans"].cluster_centers_
    scaler = final_pipeline.named_steps["preprocess"].named_steps["scaler"]
    centers_before_scaling = scaler.inverse_transform(centers_scaled)
    centroids = pd.DataFrame(centers_before_scaling, columns=HOUR_COLUMNS)
    centroids.insert(0, "cluster", range(selected_k))
    centroids.to_csv(CENTROIDS_PATH, index=False)

    weekday_weekend_labels = test_frame["is_weekend"].to_numpy()
    baseline_weekday_silhouette = safe_silhouette(
        X_test_transformed, weekday_weekend_labels
    )
    global_center = X_test_transformed.mean(axis=0, keepdims=True)
    baseline_global_distance = float(
        np.linalg.norm(X_test_transformed - global_center, axis=1).mean()
    )

    data_quality = {}
    if QUALITY_PATH.exists():
        data_quality = json.loads(QUALITY_PATH.read_text(encoding="utf-8"))

    summary = {
        "data_source": data_quality.get("source_file", "unknown"),
        "is_demo_data": data_quality.get("source_file") == "synthetic_demo",
        "selected_mode": selected_mode,
        "selected_k": selected_k,
        "selection_validation_silhouette": float(
            selected["validation_silhouette_mean"]
        ),
        "selection_stability_ari": float(selected["stability_ari_mean"]),
        "seed_variability": {
            "number_of_seeds": int(selected["number_of_seeds"]),
            "validation_silhouette_std": float(selected["validation_silhouette_std"]),
            "validation_silhouette_min": float(selected["validation_silhouette_min"]),
            "validation_silhouette_max": float(selected["validation_silhouette_max"]),
            "stability_ari_std": float(selected["stability_ari_std"]),
            "stability_ari_min": float(selected["stability_ari_min"]),
            "stability_ari_max": float(selected["stability_ari_max"]),
        },
        "test_silhouette": test_silhouette,
        "test_mean_distance_to_centroid": test_distance,
        "baseline_weekday_weekend_test_silhouette": baseline_weekday_silhouette,
        "baseline_global_mean_test_distance": baseline_global_distance,
        "split": {
            "train_days": len(train_frame),
            "validation_days": len(validation_frame),
            "test_days": len(test_frame),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "validation_end": validation_end.strftime("%Y-%m-%d"),
            "test_start": test_frame["date"].min().strftime("%Y-%m-%d"),
            "test_end": test_frame["date"].max().strftime("%Y-%m-%d"),
        },
        "test_used_for_model_selection": False,
        "feature_count": len(HOUR_COLUMNS),
        "artifact_variant": ARTIFACT_VARIANT,
        "model_path": str(MODEL_PATH.relative_to(MODEL_PATH.parents[2])),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("HUAN LUYEN HOAN THANH")
    print(f"Che do da chon: {selected_mode}")
    print(f"K da chon: {selected_k}")
    print(f"Test silhouette: {test_silhouette:.4f}")
    print(f"Model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
