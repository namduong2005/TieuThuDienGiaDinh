from pathlib import Path
import sys


def configure_console() -> None:
    """Allow Vietnamese paths to be printed safely in Windows PowerShell."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


configure_console()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
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

HOUR_COLUMNS = [f"hour_{hour:02d}" for hour in range(24)]


def ensure_directories() -> None:
    for directory in (
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
