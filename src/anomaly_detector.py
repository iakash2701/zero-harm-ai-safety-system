import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.utils import MODEL_DIR, PROCESSED_DIR, REPORT_DIR, make_project_dirs


SENSOR_COLUMNS = ["temperature", "gas_level", "vibration", "pressure", "noise_level"]


def run_anomaly_detection():
    make_project_dirs()
    df = pd.read_csv(PROCESSED_DIR / "final_safety_dataset.csv")
    sensor_data = df[SENSOR_COLUMNS].copy()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sensor_data)

    model = IsolationForest(contamination=0.06, random_state=42)
    anomaly_flag = model.fit_predict(scaled_data)
    scores = model.decision_function(scaled_data)

    result = df[["timestamp", "zone_id", "worker_name", "risk_label"] + SENSOR_COLUMNS].copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = anomaly_flag == -1
    result = result.sort_values(["is_anomaly", "anomaly_score"], ascending=[False, True])
    result.to_csv(REPORT_DIR / "anomaly_report.csv", index=False)

    joblib.dump({"model": model, "scaler": scaler, "columns": SENSOR_COLUMNS}, MODEL_DIR / "anomaly_detector.pkl")
    return REPORT_DIR / "anomaly_report.csv"


if __name__ == "__main__":
    print(run_anomaly_detection())
