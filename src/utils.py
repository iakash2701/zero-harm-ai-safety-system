from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"


def make_project_dirs():
    """Create folders used by the project if they are missing."""
    folders = [
        DATA_DIR / "raw",
        SAMPLE_DIR,
        PROCESSED_DIR,
        MODEL_DIR,
        REPORT_DIR,
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def get_risk_level(score):
    if score >= 85:
        return "Critical"
    if score >= 65:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"
