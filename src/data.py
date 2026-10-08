import argparse
import json
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DAILY_PROFILES_PATH,
    FALLBACK_ZIP_URLS,
    QUALITY_PATH,
    RAW_TEXT_PATH,
    RAW_ZIP_PATH,
    UCI_ZIP_URL,
    ensure_directories,
)


def download_dataset(force: bool = False) -> Path:
    ensure_directories()
    if RAW_TEXT_PATH.exists() and not force:
        print(f"Du lieu tho da ton tai: {RAW_TEXT_PATH}")
        return RAW_TEXT_PATH

    zip_is_valid = RAW_ZIP_PATH.exists() and zipfile.is_zipfile(RAW_ZIP_PATH)
    if force or not zip_is_valid:
        if RAW_ZIP_PATH.exists():
            RAW_ZIP_PATH.unlink()

        download_errors = []
        for source_number, source_url in enumerate(
            [*FALLBACK_ZIP_URLS, UCI_ZIP_URL], start=1
        ):
            print(f"Dang tai du lieu, nguon {source_number}...")
            try:
                request = urllib.request.Request(
                    source_url,
                    headers={"User-Agent": "Project24-Electricity-Clustering/1.0"},
                )
                with urllib.request.urlopen(request, timeout=15) as response:
                    with RAW_ZIP_PATH.open("wb") as destination:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            destination.write(block)
                if not zipfile.is_zipfile(RAW_ZIP_PATH):
                    raise RuntimeError("Tep tai ve khong phai ZIP hop le")
                print(f"Tai thanh cong tu nguon {source_number}.")
                break
            except (urllib.error.URLError, OSError, RuntimeError) as error:
                download_errors.append(f"Nguon {source_number}: {error}")
                if RAW_ZIP_PATH.exists():
                    RAW_ZIP_PATH.unlink()
        else:
            details = " | ".join(download_errors)
            raise RuntimeError(
                "Khong tai duoc du lieu tu UCI hoac cac nguon sao luu. "
                f"Chi tiet: {details}. Hay dat ZIP thu cong vao {RAW_ZIP_PATH}."
            )

    print("Dang giai nen du lieu...")
    with zipfile.ZipFile(RAW_ZIP_PATH) as archive:
        members = archive.namelist()
        target = next(
            (name for name in members if name.endswith("household_power_consumption.txt")),
            None,
        )
        if target is None:
            raise FileNotFoundError("Khong tim thay household_power_consumption.txt trong ZIP")
        with archive.open(target) as source, RAW_TEXT_PATH.open("wb") as destination:
            destination.write(source.read())

    print(f"Da luu du lieu tho: {RAW_TEXT_PATH}")
    return RAW_TEXT_PATH


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def prepare_dataset(
    input_path: Path = RAW_TEXT_PATH,
    output_path: Path = DAILY_PROFILES_PATH,
    max_missing_hours: int = 2,
    chunk_size: int = 250_000,
) -> Path:
    ensure_directories()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay {input_path}. Hay chay voi --download truoc."
        )

    hourly_parts = []
    total_rows = 0
    missing_power_rows = 0
    invalid_datetime_rows = 0

    reader = pd.read_csv(
        input_path,
        sep=";",
        usecols=["Date", "Time", "Global_active_power"],
        dtype=str,
        na_values=["?"],
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)
        timestamps = pd.to_datetime(
            chunk["Date"].fillna("") + " " + chunk["Time"].fillna(""),
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )
        power = pd.to_numeric(chunk["Global_active_power"], errors="coerce")
        missing_power_rows += int(power.isna().sum())
        invalid_datetime_rows += int(timestamps.isna().sum())

        valid = timestamps.notna() & power.notna()
        frame = pd.DataFrame(
            {
                "timestamp": timestamps.loc[valid],
                "power": power.loc[valid].astype(float),
            }
        )
        frame["hour"] = frame["timestamp"].dt.floor("h")
        hourly_parts.append(frame.groupby("hour")["power"].agg(["sum", "count"]))
        print(f"Da doc chunk {chunk_number}: {total_rows:,} dong")

    combined = pd.concat(hourly_parts).groupby(level=0)[["sum", "count"]].sum()
    hourly = combined["sum"] / combined["count"]
    full_hours = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(full_hours)

    hourly_frame = pd.DataFrame({"power": hourly})
    hourly_frame["date"] = hourly_frame.index.normalize()
    hourly_frame["hour"] = hourly_frame.index.hour

    daily = hourly_frame.pivot(index="date", columns="hour", values="power")
    daily = daily.reindex(columns=range(24))
    missing_hours = daily.isna().sum(axis=1)
    keep_mask = missing_hours <= max_missing_hours
    kept = daily.loc[keep_mask].copy()

    kept.columns = [f"hour_{hour:02d}" for hour in kept.columns]
    kept.insert(0, "date", kept.index.strftime("%Y-%m-%d"))
    kept["missing_hours"] = missing_hours.loc[keep_mask].to_numpy()
    dates = pd.to_datetime(kept["date"])
    kept["day_of_week"] = dates.dt.day_name()
    kept["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    kept["month"] = dates.dt.month
    kept["season"] = kept["month"].map(season_from_month)
    kept.reset_index(drop=True).to_csv(output_path, index=False)

    quality = {
        "source_file": str(input_path),
        "total_raw_rows": total_rows,
        "missing_power_rows": missing_power_rows,
        "invalid_datetime_rows": invalid_datetime_rows,
        "total_calendar_days": int(len(daily)),
        "kept_days": int(keep_mask.sum()),
        "removed_days": int((~keep_mask).sum()),
        "max_missing_hours_allowed": max_missing_hours,
    }
    QUALITY_PATH.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(f"Da tao: {output_path}")
    print(json.dumps(quality, indent=2))
    return output_path


def generate_demo_dataset(
    output_path: Path = DAILY_PROFILES_PATH,
    number_of_days: int = 240,
    random_state: int = 42,
) -> Path:
    """Create reproducible daily profiles so the whole pipeline can be tested."""

    ensure_directories()
    rng = np.random.default_rng(random_state)
    dates = pd.date_range("2009-01-01", periods=number_of_days, freq="D")
    hours = np.arange(24)

    evening_peak = 0.45 + 1.55 * np.exp(-0.5 * ((hours - 20) / 2.2) ** 2)
    daytime_peak = 0.65 + 1.15 * np.exp(-0.5 * ((hours - 13) / 3.4) ** 2)
    double_peak = (
        0.40
        + 0.85 * np.exp(-0.5 * ((hours - 7) / 1.8) ** 2)
        + 1.20 * np.exp(-0.5 * ((hours - 19) / 2.2) ** 2)
    )
    patterns = np.vstack([evening_peak, daytime_peak, double_peak])

    rows = []
    for date in dates:
        if date.dayofweek >= 5:
            pattern_id = rng.choice([1, 2], p=[0.7, 0.3])
        else:
            pattern_id = rng.choice([0, 2], p=[0.7, 0.3])
        seasonal_scale = 1.12 if date.month in (12, 1, 2) else 1.0
        values = patterns[pattern_id] * seasonal_scale
        values = np.clip(values + rng.normal(0, 0.09, size=24), 0.05, None)

        missing_count = int(rng.choice([0, 0, 0, 1, 2]))
        if missing_count:
            missing_positions = rng.choice(24, size=missing_count, replace=False)
            values[missing_positions] = np.nan

        row = {"date": date.strftime("%Y-%m-%d")}
        row.update({f"hour_{hour:02d}": values[hour] for hour in range(24)})
        row.update(
            {
                "missing_hours": missing_count,
                "day_of_week": date.day_name(),
                "is_weekend": int(date.dayofweek >= 5),
                "month": date.month,
                "season": season_from_month(date.month),
                "demo_true_pattern": int(pattern_id),
            }
        )
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    quality = {
        "source_file": "synthetic_demo",
        "kept_days": number_of_days,
        "note": "Chi dung de kiem tra code, khong dung lam ket qua bao cao.",
    }
    QUALITY_PATH.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(f"Da tao du lieu demo: {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Tai va xu ly du lieu dien UCI")
    parser.add_argument("--download", action="store_true", help="Tai du lieu UCI")
    parser.add_argument("--prepare", action="store_true", help="Tao daily_profiles.csv")
    parser.add_argument("--demo", action="store_true", help="Tao du lieu mo phong de chay thu")
    parser.add_argument("--force", action="store_true", help="Tai lai tep ZIP")
    parser.add_argument("--max-missing-hours", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    if not (args.download or args.prepare or args.demo):
        raise SystemExit("Hay chon --demo hoac --download --prepare")
    try:
        if args.demo:
            generate_demo_dataset()
            return
        if args.download:
            download_dataset(force=args.force)
        if args.prepare:
            prepare_dataset(max_missing_hours=args.max_missing_hours)
    except (RuntimeError, FileNotFoundError, zipfile.BadZipFile) as error:
        print("\nKHONG THE CHUAN BI DU LIEU THAT")
        print(str(error))
        print("Trang tai chinh thuc:")
        print(
            "https://archive.ics.uci.edu/dataset/235/"
            "individual+household+electric+power+consumption"
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
