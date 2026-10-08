import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    ASSIGNMENTS_PATH,
    CENTROIDS_PATH,
    EXPERIMENTS_PATH,
    FIGURES_DIR,
    HOUR_COLUMNS,
    REPORTS_DIR,
    SUMMARY_PATH,
    ensure_directories,
)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

MODE_LABELS = {
    "absolute": "Mức tuyệt đối",
    "shape": "Chuẩn hóa hình dạng",
}


def save_metric_plot(experiments: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for mode, group in experiments.groupby("mode"):
        group = group.sort_values("k")
        label = MODE_LABELS.get(mode, mode)
        axes[0].plot(group["k"], group["validation_silhouette_mean"], marker="o", label=label)
        axes[1].plot(group["k"], group["train_inertia_per_day_mean"], marker="o", label=label)
        axes[2].plot(group["k"], group["stability_ari_mean"], marker="o", label=label)

    axes[0].set_title("Silhouette trên tập validation")
    axes[1].set_title("Inertia trung bình mỗi ngày trên train")
    axes[2].set_title("Độ ổn định giữa các seed")
    for axis in axes:
        axis.set_xlabel("Số cụm K")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "01_chon_k.png", dpi=180)
    plt.close(figure)


def save_centroid_plot(centroids: pd.DataFrame, mode: str) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.5))
    hours = np.arange(24)
    for _, row in centroids.iterrows():
        axis.plot(hours, row[HOUR_COLUMNS].astype(float), marker="o", label=f"Cụm {int(row['cluster'])}")
    axis.set_xticks(hours)
    axis.set_xlabel("Giờ trong ngày")
    axis.set_ylabel("Mức tương đối" if mode == "shape" else "Công suất tác dụng trung bình (kW)")
    axis.set_title("Tâm cụm theo hồ sơ tiêu thụ điện 24 giờ")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "02_tam_cum_24_gio.png", dpi=180)
    plt.close(figure)


def save_cluster_size_plot(assignments: pd.DataFrame) -> None:
    counts = assignments["cluster"].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(counts.index.astype(str), counts.values, color="#27577f")
    axis.set_xlabel("Cụm")
    axis.set_ylabel("Số ngày")
    axis.set_title("Số ngày trong từng cụm")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "03_kich_thuoc_cum.png", dpi=180)
    plt.close(figure)


def save_weekday_profile(assignments: pd.DataFrame) -> None:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    table = pd.crosstab(assignments["day_of_week"], assignments["cluster"], normalize="index")
    table = table.reindex(order).fillna(0)
    table.index = ["Thứ hai", "Thứ ba", "Thứ tư", "Thứ năm", "Thứ sáu", "Thứ bảy", "Chủ nhật"]
    figure, axis = plt.subplots(figsize=(10, 5))
    table.plot(kind="bar", stacked=True, ax=axis, colormap="tab10")
    axis.set_xlabel("Ngày trong tuần")
    axis.set_ylabel("Tỷ lệ")
    axis.set_title("Phân bố cụm theo ngày trong tuần")
    axis.legend(title="Cụm", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "04_cum_theo_thu.png", dpi=180)
    plt.close(figure)


def save_calendar_plot(assignments: pd.DataFrame) -> None:
    data = assignments.sort_values("date").copy()
    dates = pd.to_datetime(data["date"])
    start_monday = dates.min() - pd.Timedelta(days=int(dates.min().dayofweek))
    data["week_index"] = ((dates - start_monday).dt.days // 7).astype(int)
    data["weekday_index"] = dates.dt.dayofweek

    figure, axis = plt.subplots(figsize=(14, 4.5))
    scatter = axis.scatter(
        data["week_index"],
        data["weekday_index"],
        c=data["cluster"],
        cmap="tab10",
        marker="s",
        s=65,
    )
    axis.set_yticks(range(7), ["T2", "T3", "T4", "T5", "T6", "T7", "CN"])
    axis.invert_yaxis()
    axis.set_xlabel("Số tuần tính từ đầu bộ dữ liệu")
    axis.set_title("Lịch phân cụm theo ngày")
    figure.colorbar(scatter, ax=axis, label="Cụm")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "05_lich_phan_cum.png", dpi=180)
    plt.close(figure)


def write_model_card(summary: dict, assignments: pd.DataFrame) -> None:
    cluster_counts = assignments["cluster"].value_counts().sort_index()
    counts_text = "\n".join(
        f"- Cụm {cluster}: {count} ngày" for cluster, count in cluster_counts.items()
    )
    demo_warning = (
        "**Warning: this artifact was trained on synthetic demo data and must not be used "
        "as the final project result.**\n"
        if summary.get("is_demo_data")
        else ""
    )
    content = f"""# Thẻ mô hình

{demo_warning}

## Mục đích

Phân nhóm các ngày sử dụng điện của hộ gia đình dựa trên hồ sơ công suất 24 giờ. Mô hình chỉ mang tính mô tả, không chứng minh quan hệ nhân quả và không đại diện cho hành vi của mọi hộ gia đình.

## Cấu hình được chọn

- Chế độ tiền xử lý: `{summary['selected_mode']}`
- Số cụm: `{summary['selected_k']}`
- Nguồn dữ liệu: `{summary.get('data_source', 'unknown')}`
- Silhouette trên test: `{summary['test_silhouette']:.4f}`
- Khoảng cách trung bình tới tâm cụm trên test: `{summary['test_mean_distance_to_centroid']:.4f}`
- Có dùng test để chọn mô hình: `{summary['test_used_for_model_selection']}`

## Kích thước cụm

{counts_text}

## Giới hạn

- Dữ liệu chỉ từ một hộ tại Pháp, không được khái quát cho toàn bộ dân số.
- Số hiệu cụm chỉ là mã định danh, không phải thứ hạng.
- Chuẩn hóa hình dạng sẽ loại bỏ thông tin về mức tiêu thụ tuyệt đối nếu được chọn.
- Chỉ giữ các ngày thiếu không quá ngưỡng công bố; giá trị thiếu được điền bằng tham số học từ train.
- Mô hình mô tả độ tương đồng, không chứng minh quan hệ nhân quả.
"""
    (REPORTS_DIR / "model_card.md").write_text(content, encoding="utf-8")


def main():
    ensure_directories()
    required = [EXPERIMENTS_PATH, ASSIGNMENTS_PATH, CENTROIDS_PATH, SUMMARY_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Chua co ket qua train: {missing}")

    experiments = pd.read_csv(EXPERIMENTS_PATH)
    assignments = pd.read_csv(ASSIGNMENTS_PATH, parse_dates=["date"])
    centroids = pd.read_csv(CENTROIDS_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    save_metric_plot(experiments)
    save_centroid_plot(centroids, summary["selected_mode"])
    save_cluster_size_plot(assignments)
    save_weekday_profile(assignments)
    save_calendar_plot(assignments)
    write_model_card(summary, assignments)

    report_lines = [
        "PROJECT 24 - TÓM TẮT KẾT QUẢ HỌC MÁY",
        f"Nguồn dữ liệu: {summary.get('data_source', 'unknown')}",
        f"Dữ liệu demo: {summary.get('is_demo_data', False)}",
        f"Tiền xử lý được chọn: {summary['selected_mode']}",
        f"K được chọn: {summary['selected_k']}",
        f"Silhouette validation: {summary['selection_validation_silhouette']:.4f}",
        f"Độ ổn định ARI: {summary['selection_stability_ari']:.4f}",
        f"Silhouette test: {summary['test_silhouette']:.4f}",
        f"Khoảng cách trung bình trên test: {summary['test_mean_distance_to_centroid']:.4f}",
        f"Silhouette baseline ngày thường/cuối tuần: {summary['baseline_weekday_weekend_test_silhouette']:.4f}",
    ]
    (REPORTS_DIR / "report_summary.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("DA TAO BAO CAO VA BIEU DO")
    print(f"Thu muc: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
