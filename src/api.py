import json
from functools import lru_cache
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.config import (
    ANOMALIES_PATH,
    ASSIGNMENTS_PATH,
    DAILY_PROFILES_PATH,
    EDA_SUMMARY_PATH,
    EXPERIMENTS_PATH,
    HOUR_COLUMNS,
    MODEL_PATH,
    PROJECT_ROOT,
    QUALITY_PATH,
    REMOVED_DAYS_PATH,
    SEASON_PROFILE_PATH,
    SUMMARY_PATH,
)


class PredictionRequest(BaseModel):
    hourly_power: list[float | None] = Field(
        min_length=24,
        max_length=24,
        description=(
            "24 giá trị Global_active_power trung bình từ giờ 00 đến 23, "
            "đơn vị kW. Có thể để trống tối đa 2 giờ."
        ),
        examples=[[
            0.45, 0.42, 0.39, 0.38, 0.37, 0.40,
            0.55, 0.82, 0.76, 0.63, 0.58, 0.56,
            0.54, 0.53, 0.55, 0.61, 0.72, 0.95,
            1.35, 1.72, 1.84, 1.50, 1.02, 0.68,
        ]],
    )

    @field_validator("hourly_power")
    @classmethod
    def validate_hourly_power(cls, values):
        if sum(value is None for value in values) > 2:
            raise ValueError("Chỉ được thiếu tối đa 2 giờ trong một ngày")
        for value in values:
            if value is None:
                continue
            if not np.isfinite(value):
                raise ValueError("Giá trị công suất phải là số hữu hạn")
            if value < 0:
                raise ValueError("Công suất không được là số âm")
        return values


class PredictionResponse(BaseModel):
    cluster: int
    cluster_label: str
    distance_to_centroid: float
    anomaly_threshold: float | None
    is_anomaly: bool | None
    daily_mean_power_kw: float
    daily_peak_power_kw: float
    imputed_hours: list[int]
    centroid_profile: list[float]
    outside_training_domain: bool
    warnings: list[str]
    training_domain_kw: dict[str, float]


class ModelService:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Chưa có mô hình. Hãy chạy run_real.bat hoặc py -m src.train trước."
            )
        self.pipeline = joblib.load(MODEL_PATH)
        if not {"preprocess", "kmeans"}.issubset(self.pipeline.named_steps):
            raise ValueError("Tệp mô hình không đúng cấu trúc pipeline của dự án")

        self.summary = self._read_json(SUMMARY_PATH)
        self.thresholds = self._load_anomaly_thresholds()
        self.centroids = self._centroids_in_report_units()
        self.cluster_labels = self._build_cluster_labels()
        self.daily_profiles = self._load_csv(DAILY_PROFILES_PATH, "daily_profiles.csv")
        self.assignments = self._load_csv(ASSIGNMENTS_PATH, "cluster_assignments.csv")
        self.day_records = self._build_day_records()
        self.training_domain = self._calculate_training_domain()

    @staticmethod
    def _read_json(path):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @staticmethod
    def _load_csv(path, label):
        if not path.exists():
            raise FileNotFoundError(f"Chưa có {label}. Hãy chạy run_real.bat trước.")
        frame = pd.read_csv(path)
        if "date" in frame:
            frame["date"] = frame["date"].astype(str)
        return frame

    @staticmethod
    def _load_anomaly_thresholds():
        if not ANOMALIES_PATH.exists():
            return {}
        anomalies = pd.read_csv(ANOMALIES_PATH)
        required = {"cluster", "anomaly_threshold"}
        if not required.issubset(anomalies.columns):
            return {}
        return {
            int(cluster): float(threshold)
            for cluster, threshold in anomalies.groupby("cluster")[
                "anomaly_threshold"
            ].first().items()
        }

    def _centroids_in_report_units(self):
        centers = self.pipeline.named_steps["kmeans"].cluster_centers_
        scaler = self.pipeline.named_steps["preprocess"].named_steps["scaler"]
        return scaler.inverse_transform(centers)

    def _build_cluster_labels(self):
        order = np.argsort(self.centroids.mean(axis=1))
        if len(order) == 1:
            return {int(order[0]): "Mức tiêu thụ duy nhất"}
        labels = {}
        for position, cluster in enumerate(order):
            if position == 0:
                label = "Mức tiêu thụ thấp"
            elif position == len(order) - 1:
                label = "Mức tiêu thụ cao"
            else:
                label = "Mức tiêu thụ trung bình"
            labels[int(cluster)] = label
        return labels

    def _build_day_records(self):
        assignment_columns = [
            column
            for column in ["date", "split", "cluster", "day_of_week", "season"]
            if column in self.assignments
        ]
        records = self.daily_profiles.merge(
            self.assignments[assignment_columns],
            on="date",
            how="inner",
            suffixes=("", "_assignment"),
        )
        if ANOMALIES_PATH.exists():
            anomalies = pd.read_csv(ANOMALIES_PATH)
            anomalies["date"] = anomalies["date"].astype(str)
            records = records.merge(
                anomalies[["date", "distance_to_centroid", "is_anomaly"]],
                on="date",
                how="left",
            )
        else:
            records["distance_to_centroid"] = np.nan
            records["is_anomaly"] = False
        records["is_anomaly"] = records["is_anomaly"].fillna(False).astype(bool)
        return records

    def _calculate_training_domain(self):
        reference = self.day_records[
            self.day_records["split"].isin(["train", "validation"])
        ][HOUR_COLUMNS].to_numpy(dtype=float)
        return {
            "minimum": float(np.nanmin(reference)),
            "maximum": float(np.nanmax(reference)),
        }

    def predict(self, values: list[float | None]) -> PredictionResponse:
        numeric_values = np.array(
            [np.nan if value is None else float(value) for value in values],
            dtype=float,
        )
        frame = pd.DataFrame([numeric_values], columns=HOUR_COLUMNS)
        transformed = self.pipeline.named_steps["preprocess"].transform(frame)
        model = self.pipeline.named_steps["kmeans"]
        cluster = int(model.predict(transformed)[0])
        distance = float(model.transform(transformed)[0, cluster])
        threshold = self.thresholds.get(cluster)
        is_anomaly = None if threshold is None else distance > threshold
        missing_hours = [index for index, value in enumerate(values) if value is None]
        observed_min = float(np.nanmin(numeric_values))
        observed_max = float(np.nanmax(numeric_values))
        outside_domain = (
            observed_min < self.training_domain["minimum"]
            or observed_max > self.training_domain["maximum"]
        )
        warnings = []
        if missing_hours:
            warnings.append(
                "Các giờ bị thiếu được điền bằng trung vị học từ train: "
                + ", ".join(f"{hour:02d}h" for hour in missing_hours)
            )
        if outside_domain:
            warnings.append(
                "Có giá trị nằm ngoài miền công suất theo giờ của train và validation; "
                "kết quả cần được xem thận trọng."
            )

        return PredictionResponse(
            cluster=cluster,
            cluster_label=self.cluster_labels[cluster],
            distance_to_centroid=distance,
            anomaly_threshold=threshold,
            is_anomaly=is_anomaly,
            daily_mean_power_kw=float(np.nanmean(numeric_values)),
            daily_peak_power_kw=float(np.nanmax(numeric_values)),
            imputed_hours=missing_hours,
            centroid_profile=[float(value) for value in self.centroids[cluster]],
            outside_training_domain=outside_domain,
            warnings=warnings,
            training_domain_kw=self.training_domain,
        )

    def model_info(self):
        return {
            "algorithm": "K-means",
            "feature_count": len(HOUR_COLUMNS),
            "selected_k": int(self.pipeline.named_steps["kmeans"].n_clusters),
            "selected_mode": self.summary.get("selected_mode", "unknown"),
            "test_silhouette": self.summary.get("test_silhouette"),
            "data_source": self.summary.get("data_source", "unknown"),
            "cluster_labels": self.cluster_labels,
            "input_unit": "kW - công suất tác dụng trung bình theo giờ",
            "training_domain_kw": self.training_domain,
        }

    def list_days(self, season: str | None = None):
        frame = self.day_records
        if season:
            frame = frame[frame["season"] == season]
        return [
            {
                "date": row.date,
                "cluster": int(row.cluster),
                "cluster_label": self.cluster_labels[int(row.cluster)],
                "season": row.season,
                "day_of_week": row.day_of_week,
                "split": row.split,
                "is_anomaly": bool(row.is_anomaly),
            }
            for row in frame.itertuples()
        ]

    def day_profile(self, date: str):
        matches = self.day_records[self.day_records["date"] == date]
        if matches.empty:
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy ngày này hoặc ngày đã bị loại do thiếu dữ liệu.",
            )
        row = matches.iloc[0]
        hourly_values = [
            None if pd.isna(row[column]) else float(row[column]) for column in HOUR_COLUMNS
        ]
        observed_values = [value for value in hourly_values if value is not None]
        cluster = int(row["cluster"])
        return {
            "date": date,
            "cluster": cluster,
            "cluster_label": self.cluster_labels[cluster],
            "season": str(row["season"]),
            "day_of_week": str(row["day_of_week"]),
            "split": str(row["split"]),
            "missing_hours": int(row.get("missing_hours", 0)),
            "hourly_power": hourly_values,
            "centroid_profile": [float(value) for value in self.centroids[cluster]],
            "distance_to_centroid": (
                None
                if pd.isna(row["distance_to_centroid"])
                else float(row["distance_to_centroid"])
            ),
            "is_anomaly": bool(row["is_anomaly"]),
            "daily_mean_power_kw": float(np.mean(observed_values)),
            "daily_peak_power_kw": float(np.max(observed_values)),
        }

    def dashboard(self):
        experiments = []
        if EXPERIMENTS_PATH.exists():
            raw_experiments = pd.read_csv(EXPERIMENTS_PATH)
            for k in sorted(raw_experiments["k"].unique()):
                row = {"k": int(k)}
                for mode in ("absolute", "shape"):
                    match = raw_experiments[
                        (raw_experiments["k"] == k)
                        & (raw_experiments["mode"] == mode)
                    ]
                    value = (
                        match.iloc[0]["validation_silhouette_mean"]
                        if not match.empty
                        else np.nan
                    )
                    row[f"{mode}_silhouette"] = (
                        None if pd.isna(value) else float(value)
                    )
                experiments.append(row)
        season_profile = []
        if SEASON_PROFILE_PATH.exists():
            raw_seasons = pd.read_csv(SEASON_PROFILE_PATH)
            for item in raw_seasons.itertuples(index=False):
                for cluster in range(len(self.centroids)):
                    season_profile.append({
                        "season": item.season,
                        "cluster": cluster,
                        "share": float(getattr(item, f"_{cluster + 1}")),
                    })
        normalized_summary = {
            **self.summary,
            "mode": self.summary.get("selected_mode"),
            "validation_silhouette": self.summary.get(
                "selection_validation_silhouette"
            ),
            "stability_ari": self.summary.get("selection_stability_ari"),
            "random_baseline": self.summary.get(
                "baseline_weekday_weekend_test_silhouette"
            ),
        }
        cluster_sizes = [
            {"cluster": int(cluster), "count": int(count)}
            for cluster, count in self.day_records["cluster"]
            .value_counts()
            .sort_index()
            .items()
        ]
        return {
            "summary": normalized_summary,
            "quality": self._read_json(QUALITY_PATH),
            "eda": self._read_json(EDA_SUMMARY_PATH),
            "cluster_counts": {
                str(key): int(value)
                for key, value in self.day_records["cluster"].value_counts().sort_index().items()
            },
            "cluster_labels": self.cluster_labels,
            "cluster_sizes": cluster_sizes,
            "centroids": [
                {"cluster": index, "profile": [float(value) for value in profile]}
                for index, profile in enumerate(self.centroids)
            ],
            "experiments": experiments,
            "season_profile": season_profile,
            "limitations": [
                "Dữ liệu chỉ đến từ một hộ gia đình tại Pháp.",
                "Cụm mô tả độ tương đồng, không chứng minh nguyên nhân tiêu thụ điện.",
                "Silhouette test còn thấp nên các cụm có thể chồng lấn.",
                "Nhãn cụm là mã định danh, không phải thứ hạng tốt hoặc xấu.",
            ],
        }

    def data_quality(self):
        removed = (
            pd.read_csv(REMOVED_DAYS_PATH).to_dict("records")
            if REMOVED_DAYS_PATH.exists()
            else []
        )
        eda = self._read_json(EDA_SUMMARY_PATH)
        return {
            "quality": self._read_json(QUALITY_PATH),
            "eda": {
                **eda,
                "global_active_power_min": eda.get("daily_mean_kw", {}).get("min"),
                "global_active_power_max": eda.get("daily_peak_kw", {}).get("max"),
            },
            "date_range": {
                "start": eda.get("start_date"),
                "end": eda.get("end_date"),
            },
            "anomaly_count": int(eda.get("anomaly_days", 0)),
            "removed_days": removed,
        }


@lru_cache(maxsize=1)
def get_model_service():
    return ModelService()


def require_model_service():
    try:
        return get_model_service()
    except (FileNotFoundError, ValueError, OSError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


ModelServiceDependency = Annotated[ModelService, Depends(require_model_service)]

app = FastAPI(
    title="API phân cụm ngày tiêu thụ điện",
    version="2.0.0",
    description=(
        "API dự đoán và tra cứu hồ sơ công suất 24 giờ bằng mô hình K-means đã huấn luyện."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

WEB_DIR = PROJECT_ROOT / "web"
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.get("/", include_in_schema=False)
@app.get("/app", include_in_schema=False)
@app.get("/analyze", include_in_schema=False)
@app.get("/calendar", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
@app.get("/data-quality", include_in_schema=False)
def website_pages():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api", tags=["Hướng dẫn"])
def api_root():
    return {
        "message": "API phân cụm ngày tiêu thụ điện đang hoạt động",
        "website": "/",
        "documentation": "/docs",
        "day_profile": "GET /api/day-profile?date=2010-11-26",
        "prediction": "POST /predict",
    }


@app.get("/health", tags=["Hệ thống"])
def health(service: ModelServiceDependency):
    return {
        "status": "ready",
        "model_loaded": True,
        "selected_k": int(service.pipeline.named_steps["kmeans"].n_clusters),
    }


@app.get("/model-info", tags=["Mô hình"])
def model_info(service: ModelServiceDependency):
    return service.model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["Dự đoán"])
def predict(request: PredictionRequest, service: ModelServiceDependency):
    return service.predict(request.hourly_power)


@app.get("/api/day-profile", tags=["Hồ sơ ngày"])
def day_profile(
    service: ModelServiceDependency,
    date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2010-11-26"]),
):
    return service.day_profile(date)


@app.get("/api/days", tags=["Hồ sơ ngày"])
def days(
    service: ModelServiceDependency,
    season: str | None = Query(default=None),
):
    if season is not None and season not in {"winter", "spring", "summer", "autumn"}:
        raise HTTPException(status_code=422, detail="Mùa không hợp lệ")
    return {"days": service.list_days(season)}


@app.get("/api/dashboard", tags=["Đánh giá"])
def dashboard(service: ModelServiceDependency):
    return service.dashboard()


@app.get("/api/data-quality", tags=["Chất lượng dữ liệu"])
def data_quality(service: ModelServiceDependency):
    return service.data_quality()
