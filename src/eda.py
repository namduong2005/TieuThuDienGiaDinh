import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    ANOMALIES_PATH,
    ASSIGNMENTS_PATH,
    DAILY_PROFILES_PATH,
    DATA_CARD_PATH,
    DATA_DICTIONARY_CSV_PATH,
    DATA_DICTIONARY_PATH,
    EDA_SUMMARY_PATH,
    ERROR_BY_PERIOD_PATH,
    ERROR_BY_SEASON_PATH,
    FIGURES_DIR,
    HOUR_COLUMNS,
    MODEL_PATH,
    QUALITY_PATH,
    SEASON_PROFILE_PATH,
    SUB_METERING_COLUMNS,
    SUB_METERING_PROFILE_PATH,
    SUMMARY_PATH,
    ensure_directories,
)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

SEASON_LABELS = {
    "winter": "Mùa đông",
    "spring": "Mùa xuân",
    "summer": "Mùa hè",
    "autumn": "Mùa thu",
}


def load_results():
    required = [DAILY_PROFILES_PATH, ASSIGNMENTS_PATH, MODEL_PATH, SUMMARY_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Chưa có đủ kết quả để phân tích: {missing}")

    daily = pd.read_csv(DAILY_PROFILES_PATH, parse_dates=["date"])
    assignments = pd.read_csv(ASSIGNMENTS_PATH, parse_dates=["date"])
    model = joblib.load(MODEL_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    quality = (
        json.loads(QUALITY_PATH.read_text(encoding="utf-8"))
        if QUALITY_PATH.exists()
        else {}
    )
    return daily, assignments, model, summary, quality


def build_analysis_frame(daily, assignments, model):
    assignment_fields = ["date", "split", "cluster"]
    frame = daily.merge(assignments[assignment_fields], on="date", how="inner")
    frame["daily_mean_kw"] = frame[HOUR_COLUMNS].mean(axis=1, skipna=True)
    frame["daily_peak_kw"] = frame[HOUR_COLUMNS].max(axis=1, skipna=True)

    transformed = model.named_steps["preprocess"].transform(frame[HOUR_COLUMNS])
    distances = model.named_steps["kmeans"].transform(transformed)
    labels = model.named_steps["kmeans"].predict(transformed)
    frame["cluster"] = labels
    frame["distance_to_centroid"] = distances[np.arange(len(frame)), labels]

    reference = frame[frame["split"].isin(["train", "validation"])]
    thresholds = reference.groupby("cluster")["distance_to_centroid"].quantile(0.95)
    global_threshold = float(reference["distance_to_centroid"].quantile(0.95))
    frame["anomaly_threshold"] = frame["cluster"].map(thresholds).fillna(global_threshold)
    frame["is_anomaly"] = frame["distance_to_centroid"] > frame["anomaly_threshold"]
    return frame


def save_tables(frame):
    anomaly_columns = [
        "date",
        "split",
        "cluster",
        "distance_to_centroid",
        "anomaly_threshold",
        "daily_mean_kw",
        "daily_peak_kw",
        "day_of_week",
        "season",
        "is_anomaly",
    ]
    frame[anomaly_columns].sort_values(
        "distance_to_centroid", ascending=False
    ).to_csv(ANOMALIES_PATH, index=False)

    season_table = pd.crosstab(
        frame["season"], frame["cluster"], normalize="index"
    ).reindex(["winter", "spring", "summer", "autumn"]).fillna(0)
    season_table.index.name = "season"
    season_table.to_csv(SEASON_PROFILE_PATH)

    available_submeters = [column for column in SUB_METERING_COLUMNS if column in frame]
    if available_submeters:
        submeter_table = frame.groupby("cluster")[available_submeters].mean()
    else:
        submeter_table = pd.DataFrame(index=sorted(frame["cluster"].unique()))
        submeter_table.index.name = "cluster"
    submeter_table.to_csv(SUB_METERING_PROFILE_PATH)
    return season_table, submeter_table


def save_error_analysis(frame):
    def summarize(group):
        return pd.Series({
            "number_of_days": len(group),
            "mean_distance_to_centroid": group["distance_to_centroid"].mean(),
            "p95_distance_to_centroid": group["distance_to_centroid"].quantile(0.95),
            "anomaly_days": int(group["is_anomaly"].sum()),
            "anomaly_rate": group["is_anomaly"].mean(),
        })

    by_season = frame.groupby("season", observed=True).apply(
        summarize, include_groups=False
    ).reset_index()
    temporal = frame.copy()
    temporal["period"] = temporal["date"].dt.to_period("Q").astype(str)
    by_period = temporal.groupby("period", observed=True).apply(
        summarize, include_groups=False
    ).reset_index()
    by_season.to_csv(ERROR_BY_SEASON_PATH, index=False)
    by_period.to_csv(ERROR_BY_PERIOD_PATH, index=False)

    figure, axis = plt.subplots(figsize=(8, 4.8))
    plot = by_season.set_index("season").reindex(SEASON_LABELS).reset_index()
    axis.bar(
        [SEASON_LABELS[value] for value in plot["season"]],
        plot["anomaly_rate"] * 100,
        color="#d95f59",
    )
    axis.set_ylabel("Tỷ lệ ngày khác biệt (%)")
    axis.set_title("Phân tích ngày khác biệt theo mùa")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "11_loi_theo_mua.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.plot(by_period["period"], by_period["anomaly_rate"] * 100, marker="o")
    axis.set_ylabel("Tỷ lệ ngày khác biệt (%)")
    axis.set_xlabel("Quý")
    axis.set_title("Biến động ngày khác biệt theo thời gian")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "12_loi_theo_thoi_gian.png", dpi=180)
    plt.close(figure)
    return by_season, by_period


def save_eda_figures(frame, season_table, submeter_table, quality):
    eda_frame = frame[frame["split"] == "train"].copy()
    q1 = eda_frame["daily_mean_kw"].quantile(0.25)
    q3 = eda_frame["daily_mean_kw"].quantile(0.75)
    upper = q3 + 1.5 * (q3 - q1)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(eda_frame["daily_mean_kw"], bins=35, color="#277da1", alpha=0.85)
    axes[0].axvline(upper, color="#d62828", linestyle="--", label="Ngưỡng ngoại lệ IQR")
    axes[0].set_title("Phân bố công suất trung bình ngày")
    axes[0].set_xlabel("Công suất trung bình (kW)")
    axes[0].set_ylabel("Số ngày")
    axes[0].legend()
    axes[1].hist(eda_frame["daily_peak_kw"], bins=35, color="#f8961e", alpha=0.85)
    axes[1].set_title("Phân bố công suất cực đại ngày")
    axes[1].set_xlabel("Công suất cực đại theo giờ (kW)")
    axes[1].set_ylabel("Số ngày")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "06_eda_phan_bo_cong_suat.png", dpi=180)
    plt.close(figure)

    missing_by_hour = eda_frame[HOUR_COLUMNS].isna().sum()
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(range(24), missing_by_hour.values, color="#577590")
    axis.set_xticks(range(24))
    axis.set_xlabel("Giờ trong ngày")
    axis.set_ylabel("Số ngày thiếu dữ liệu")
    axis.set_title("Số giá trị thiếu ở từng giờ trước khi điền")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "07_eda_missing_theo_gio.png", dpi=180)
    plt.close(figure)

    season_plot = season_table.copy()
    season_plot.index = [SEASON_LABELS.get(value, value) for value in season_plot.index]
    figure, axis = plt.subplots(figsize=(9, 5))
    season_plot.plot(kind="bar", stacked=True, ax=axis, colormap="tab10")
    axis.set_xlabel("Mùa")
    axis.set_ylabel("Tỷ lệ ngày")
    axis.set_title("Tỷ lệ các cụm theo mùa")
    axis.legend(title="Cụm", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "08_cum_theo_mua.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    if not submeter_table.empty and len(submeter_table.columns):
        renamed = submeter_table.rename(
            columns={
                "sub_metering_1_kwh": "Khu bếp (1)",
                "sub_metering_2_kwh": "Phòng giặt (2)",
                "sub_metering_3_kwh": "Bình nóng lạnh/điều hòa (3)",
            }
        )
        renamed.plot(kind="bar", ax=axis)
        axis.set_ylabel("Điện năng trung bình mỗi ngày (kWh)")
        axis.set_xlabel("Cụm")
        axis.legend(title="Nhóm sub-metering")
    else:
        axis.text(0.5, 0.5, "Không có dữ liệu sub-metering", ha="center", va="center")
        axis.set_axis_off()
    axis.set_title("Mức sub-metering trung bình theo cụm")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "09_sub_metering_theo_cum.png", dpi=180)
    plt.close(figure)

    top = frame.nlargest(15, "distance_to_centroid").sort_values(
        "distance_to_centroid"
    )
    labels = top["date"].dt.strftime("%Y-%m-%d")
    colors = np.where(top["is_anomaly"], "#d62828", "#577590")
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(labels, top["distance_to_centroid"], color=colors)
    axis.set_xlabel("Khoảng cách tới tâm cụm")
    axis.set_ylabel("Ngày")
    axis.set_title("15 ngày khác biệt nhất so với tâm cụm")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "10_ngay_bat_thuong.png", dpi=180)
    plt.close(figure)


def write_data_dictionary():
    rows = [
        ("Date", "raw", "Ngày đo", "DD/MM/YYYY", "định danh thời gian", "Không; dòng ngày lỗi bị loại"),
        ("Time", "raw", "Thời điểm đo", "HH:MM:SS", "định danh thời gian", "Không; dòng thời gian lỗi bị loại"),
        ("Global_active_power", "raw", "Công suất tác dụng trung bình phút", "kW", "nguồn tạo 24 feature", "? được coi là thiếu"),
        ("Global_reactive_power", "raw", "Công suất phản kháng", "kW", "không dùng", "Không nạp vào pipeline"),
        ("Voltage", "raw", "Điện áp trung bình phút", "V", "không dùng", "Không nạp vào pipeline"),
        ("Global_intensity", "raw", "Cường độ dòng điện", "A", "không dùng", "Không nạp vào pipeline"),
        ("Sub_metering_1", "raw", "Điện năng khu bếp", "Wh/phút", "mô tả sau phân cụm", "Cộng theo ngày, đổi sang kWh"),
        ("Sub_metering_2", "raw", "Điện năng phòng giặt", "Wh/phút", "mô tả sau phân cụm", "Cộng theo ngày, đổi sang kWh"),
        ("Sub_metering_3", "raw", "Điện năng bình nóng lạnh/điều hòa", "Wh/phút", "mô tả sau phân cụm", "Cộng theo ngày, đổi sang kWh"),
        ("date", "derived", "Ngày quan sát", "YYYY-MM-DD", "ID", "Một dòng cho mỗi ngày"),
        ("hour_00…hour_23", "derived", "Công suất trung bình từng giờ", "kW", "24 feature học máy", "Ngày thiếu >2 giờ bị loại; còn lại impute trung vị train"),
        ("missing_hours", "derived", "Số giờ thiếu trong ngày", "giờ", "metadata", "Miền 0–2 đối với ngày được giữ"),
        ("day_of_week", "derived", "Ngày trong tuần", "category", "phân tích/baseline", "Không dùng huấn luyện K-means"),
        ("is_weekend", "derived", "Cờ cuối tuần", "0/1", "baseline", "Không dùng huấn luyện K-means"),
        ("month", "derived", "Tháng", "1–12", "phân tích", "Không dùng huấn luyện K-means"),
        ("season", "derived", "Mùa khí tượng", "winter/spring/summer/autumn", "phân tích", "Suy ra từ tháng"),
        ("split", "model output", "Tập dữ liệu", "train/validation/test", "kiểm soát đánh giá", "Chia theo thời gian 60/20/20"),
        ("cluster", "model output", "Nhãn cụm K-means", "số nguyên", "đầu ra", "Mã định danh, không phải thứ hạng"),
        ("distance_to_centroid", "model output", "Khoảng cách tới tâm cụm", "không thứ nguyên", "phân tích lỗi", "Tính trong không gian đã tiền xử lý"),
        ("is_anomaly", "model output", "Ngày xa tâm cụm", "boolean", "cảnh báo", "Ngưỡng percentile 95 từ train+validation"),
    ]
    columns = ["variable", "stage", "description", "unit_or_domain", "role", "missing_or_rule"]
    dictionary = pd.DataFrame(rows, columns=columns)
    dictionary.to_csv(DATA_DICTIONARY_CSV_PATH, index=False)
    lines = ["# Data dictionary", "", "Bảng biến đầy đủ dùng trong dự án.", "", "| Biến | Giai đoạn | Mô tả | Đơn vị/miền | Vai trò | Thiếu/quy tắc |", "|---|---|---|---|---|---|"]
    lines.extend(
        f"| {row.variable} | {row.stage} | {row.description} | {row.unit_or_domain} | {row.role} | {row.missing_or_rule} |"
        for row in dictionary.itertuples(index=False)
    )
    DATA_DICTIONARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_summaries(frame, summary, quality):
    eda_frame = frame[frame["split"] == "train"].copy()
    q1 = float(eda_frame["daily_mean_kw"].quantile(0.25))
    q3 = float(eda_frame["daily_mean_kw"].quantile(0.75))
    iqr_lower = q1 - 1.5 * (q3 - q1)
    iqr_upper = q3 + 1.5 * (q3 - q1)
    iqr_count = int(
        ((eda_frame["daily_mean_kw"] < iqr_lower) | (eda_frame["daily_mean_kw"] > iqr_upper)).sum()
    )
    eda_summary = {
        "data_source": summary.get("data_source", "unknown"),
        "eda_scope": "train_only",
        "number_of_days": int(len(eda_frame)),
        "start_date": eda_frame["date"].min().strftime("%Y-%m-%d"),
        "end_date": eda_frame["date"].max().strftime("%Y-%m-%d"),
        "duplicate_date_rows": int(eda_frame["date"].duplicated().sum()),
        "missing_hour_cells_before_imputation": int(eda_frame[HOUR_COLUMNS].isna().sum().sum()),
        "days_with_missing_hours": int(eda_frame[HOUR_COLUMNS].isna().any(axis=1).sum()),
        "daily_mean_kw": eda_frame["daily_mean_kw"].describe().to_dict(),
        "daily_peak_kw": eda_frame["daily_peak_kw"].describe().to_dict(),
        "daily_mean_iqr_lower_kw": iqr_lower,
        "daily_mean_iqr_upper_kw": iqr_upper,
        "daily_mean_iqr_outlier_days_train": iqr_count,
        "anomaly_days": int(frame["is_anomaly"].sum()),
        "anomaly_rate": float(frame["is_anomaly"].mean()),
        "cluster_counts": {
            str(key): int(value) for key, value in frame["cluster"].value_counts().sort_index().items()
        },
        "season_counts": {
            str(key): int(value) for key, value in eda_frame["season"].value_counts().items()
        },
        "sub_metering_available": all(column in frame for column in SUB_METERING_COLUMNS),
    }
    EDA_SUMMARY_PATH.write_text(
        json.dumps(eda_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    split = summary.get("split", {})
    card = f"""# Thẻ dữ liệu

## Nguồn và phạm vi

- Bộ dữ liệu: Individual Household Electric Power Consumption.
- Nguồn: UCI Machine Learning Repository; DOI: 10.24432/C58K54; giấy phép CC BY 4.0.
- EDA mô tả chỉ dùng tập train trong khoảng {eda_summary['start_date']} đến {eda_summary['end_date']} để tránh nhìn trước validation/test.
- Tệp nguồn trong dự án: `{summary.get('data_source', 'unknown')}`.

## Quy mô và chất lượng

- Số dòng đo theo phút ban đầu: {quality.get('total_raw_rows', 'không có')}.
- Số ngày theo lịch: {quality.get('total_calendar_days', len(frame))}; giữ lại: {quality.get('kept_days', len(frame))}; loại: {quality.get('removed_days', 0)}.
- Dòng thiếu Global_active_power: {quality.get('missing_power_rows', 'không có')}.
- Timestamp trùng: {quality.get('duplicate_timestamp_rows', 'không có')}; công suất âm: {quality.get('negative_power_rows', 'không có')}.
- Ngày ngoại lệ theo IQR của công suất trung bình: {iqr_count}.
- Ngày bất thường theo khoảng cách tới tâm cụm: {eda_summary['anomaly_days']} ({eda_summary['anomaly_rate']:.1%}).

## Biến dùng trong học máy

- Mỗi ngày là 24 đặc trưng `hour_00` đến `hour_23`, là Global_active_power trung bình theo giờ, đơn vị kW.
- Ba biến sub-metering được đổi từ Wh sang kWh/ngày và chỉ dùng để mô tả cụm, không dùng huấn luyện.
- Ngày trong tuần, cuối tuần, tháng và mùa chỉ dùng cho baseline hoặc phân tích sau phân cụm.

## Chia dữ liệu và tiền xử lý

- Train: {split.get('train_days', 'không có')} ngày; validation: {split.get('validation_days', 'không có')} ngày; test: {split.get('test_days', 'không có')} ngày.
- Chia theo thứ tự thời gian. Imputer và scaler được học từ train khi chọn mô hình.
- Test không được dùng để chọn K hay chế độ chuẩn hóa.

## Giới hạn sử dụng

- Dữ liệu chỉ từ một hộ nên không đại diện cho mọi hộ gia đình hoặc quốc gia.
- K-means mô tả các mẫu giống nhau, không chứng minh nguyên nhân tiêu thụ điện.
- Nhãn cụm là mã định danh; cần đọc biểu đồ tâm cụm để diễn giải.
- Ngày bất thường là ngày xa tâm cụm về mặt thống kê, không mặc nhiên là lỗi hay gian lận.
"""
    DATA_CARD_PATH.write_text(card, encoding="utf-8")
    write_data_dictionary()


def main():
    ensure_directories()
    daily, assignments, model, summary, quality = load_results()
    frame = build_analysis_frame(daily, assignments, model)
    season_table, submeter_table = save_tables(frame)
    save_eda_figures(frame, season_table, submeter_table, quality)
    save_error_analysis(frame)
    write_summaries(frame, summary, quality)
    print("ĐÃ HOÀN THÀNH EDA VÀ PHÂN TÍCH SAU PHÂN CỤM")
    print(f"Số ngày bất thường: {int(frame['is_anomaly'].sum())}/{len(frame)}")
    print(f"Thư mục biểu đồ: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
