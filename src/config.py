from pathlib import Path
import os
import sys


def configure_console() -> None:
    """Allow Vietnamese paths to be printed safely in Windows PowerShell."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


configure_console()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_VARIANT = os.getenv("POWER_PROJECT_VARIANT", "real").strip().lower()
if ARTIFACT_VARIANT not in {"real", "demo"}:
    raise ValueError("POWER_PROJECT_VARIANT chỉ được là 'real' hoặc 'demo'")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed" / ARTIFACT_VARIANT
MODELS_DIR = PROJECT_ROOT / "models" / ARTIFACT_VARIANT
REPORTS_DIR = PROJECT_ROOT / "reports" / ARTIFACT_VARIANT
FIGURES_DIR = REPORTS_DIR / "figures"

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
FALLBACK_ZIP_URLS = [
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.zip",
    "https://d396qusza40orc.cloudfront.net/exdata%2Fdata%2Fhousehold_power_consumption.zip",
]
RAW_ZIP_PATH = RAW_DIR / "individual_household_electric_power_consumption.zip"
RAW_TEXT_PATH = RAW_DIR / "household_power_consumption.txt"
DAILY_PROFILES_PATH = PROCESSED_DIR / "daily_profiles.csv"

MODEL_PATH = MODELS_DIR / "kmeans_pipeline.joblib"
EXPERIMENTS_PATH = REPORTS_DIR / "experiments.csv"
ASSIGNMENTS_PATH = REPORTS_DIR / "cluster_assignments.csv"
CENTROIDS_PATH = REPORTS_DIR / "centroids.csv"
SUMMARY_PATH = REPORTS_DIR / "summary.json"
QUALITY_PATH = REPORTS_DIR / "data_quality.json"
REMOVED_DAYS_PATH = REPORTS_DIR / "removed_days.csv"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"
ANOMALIES_PATH = REPORTS_DIR / "anomalies.csv"
SEASON_PROFILE_PATH = REPORTS_DIR / "season_cluster_profile.csv"
SUB_METERING_PROFILE_PATH = REPORTS_DIR / "sub_metering_by_cluster.csv"
DATA_CARD_PATH = REPORTS_DIR / "data_card.md"
DATA_DICTIONARY_PATH = REPORTS_DIR / "data_dictionary.md"
DATA_DICTIONARY_CSV_PATH = REPORTS_DIR / "data_dictionary.csv"
ERROR_BY_SEASON_PATH = REPORTS_DIR / "error_analysis_by_season.csv"
ERROR_BY_PERIOD_PATH = REPORTS_DIR / "error_analysis_by_period.csv"

HOUR_COLUMNS = [f"hour_{hour:02d}" for hour in range(24)]
SUB_METERING_COLUMNS = [
    "sub_metering_1_kwh",
    "sub_metering_2_kwh",
    "sub_metering_3_kwh",
]


def ensure_directories() -> None:
    for directory in (
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
