import pandas as pd

from src.utils import PROCESSED_DIR, SAMPLE_DIR, make_project_dirs


def load_sample_files():
    return {
        "sensor": pd.read_csv(SAMPLE_DIR / "sensor_data.csv", parse_dates=["timestamp"]),
        "worker": pd.read_csv(SAMPLE_DIR / "worker_location.csv", parse_dates=["timestamp"]),
        "maintenance": pd.read_csv(SAMPLE_DIR / "maintenance_logs.csv", parse_dates=["last_maintenance_date", "next_due_date"]),
        "permits": pd.read_csv(SAMPLE_DIR / "work_permits.csv", parse_dates=["valid_from", "valid_to"]),
    }


def clean_sensor_data(sensor_df):
    sensor_df = sensor_df.copy()
    number_cols = ["temperature", "gas_level", "vibration", "pressure", "noise_level"]
    for col in number_cols:
        sensor_df[col] = sensor_df.groupby("zone_id")[col].transform(lambda x: x.fillna(x.median()))
    sensor_df["machine_status"] = sensor_df["machine_status"].fillna("Running")
    return sensor_df


def clean_worker_data(worker_df):
    worker_df = worker_df.copy()
    worker_df["ppe_status"] = worker_df["ppe_status"].fillna("OK")
    worker_df["entry_status"] = worker_df["entry_status"].fillna("Allowed")
    worker_df["risk_zone"] = worker_df["risk_zone"].fillna("Normal")
    return worker_df


def prepare_maintenance_data(maintenance_df):
    temp = maintenance_df.copy()
    temp["is_overdue"] = (temp["maintenance_status"] == "Overdue").astype(int)
    zone_data = temp.groupby("zone_id").agg({
        "fault_history_count": "mean",
        "is_overdue": "max",
    }).reset_index()
    zone_data["maintenance_status"] = zone_data["is_overdue"].map({1: "Overdue", 0: "OK"})
    zone_data = zone_data.drop(columns=["is_overdue"])
    return zone_data


def prepare_permit_data(permits_df):
    temp = permits_df.copy()
    temp = temp.sort_values("valid_to", ascending=False)
    return temp.drop_duplicates(["worker_id", "zone_id"], keep="first")


def preprocess_data():
    make_project_dirs()
    files = load_sample_files()
    sensor_df = clean_sensor_data(files["sensor"])
    worker_df = clean_worker_data(files["worker"])
    maintenance_df = prepare_maintenance_data(files["maintenance"])
    permits_df = prepare_permit_data(files["permits"])

    merged = sensor_df.merge(
        worker_df,
        on=["timestamp", "zone_id"],
        how="left",
    )
    merged = merged.merge(maintenance_df, on="zone_id", how="left")
    merged = merged.merge(
        permits_df[["worker_id", "zone_id", "allowed_task", "valid_from", "valid_to", "permit_status"]],
        on=["worker_id", "zone_id"],
        how="left",
    )

    merged["worker_id"] = merged["worker_id"].fillna("Unknown")
    merged["worker_name"] = merged["worker_name"].fillna("Unknown")
    merged["ppe_status"] = merged["ppe_status"].fillna("OK")
    merged["risk_zone"] = merged["risk_zone"].fillna("Normal")
    merged["entry_status"] = merged["entry_status"].fillna("Allowed")
    merged["maintenance_status"] = merged["maintenance_status"].fillna("OK")
    merged["fault_history_count"] = merged["fault_history_count"].fillna(0)
    merged["permit_status"] = merged["permit_status"].fillna("Invalid")
    merged["allowed_task"] = merged["allowed_task"].fillna("No active permit")

    invalid_time = (
        merged["valid_from"].notna() &
        merged["valid_to"].notna() &
        ((merged["timestamp"] < merged["valid_from"]) | (merged["timestamp"] > merged["valid_to"]))
    )
    merged.loc[invalid_time, "permit_status"] = "Expired"

    merged["maintenance_overdue"] = (merged["maintenance_status"] == "Overdue").astype(int)
    merged["ppe_violation"] = (merged["ppe_status"] != "OK").astype(int)
    merged["invalid_permit"] = (merged["permit_status"] != "Valid").astype(int)
    merged["restricted_entry"] = (merged["entry_status"] == "Restricted").astype(int)

    output_path = PROCESSED_DIR / "final_safety_dataset.csv"
    merged.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    print(preprocess_data())
