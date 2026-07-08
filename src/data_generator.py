from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd

from src.risk_engine import calculate_compound_risk
from src.utils import SAMPLE_DIR, make_project_dirs


ZONES = [
    "Zone-A",
    "Zone-B",
    "Zone-C",
    "Furnace Area",
    "Chemical Storage",
    "Assembly Point",
]

ZONE_COORDS = {
    "Zone-A": (17.4441, 78.3772),
    "Zone-B": (17.4452, 78.3790),
    "Zone-C": (17.4464, 78.3804),
    "Furnace Area": (17.4432, 78.3813),
    "Chemical Storage": (17.4421, 78.3785),
    "Assembly Point": (17.4471, 78.3764),
}

WORKER_NAMES = [
    "Rahul Sharma",
    "Priya Reddy",
    "Amit Kumar",
    "Sneha Patel",
    "Karan Singh",
    "Neha Verma",
    "Arjun Nair",
    "Meera Joshi",
    "Vikram Rao",
    "Anjali Das",
    "Rohit Gupta",
    "Fatima Khan",
]


def create_workers():
    workers = []
    for i, name in enumerate(WORKER_NAMES, start=1):
        workers.append({
            "worker_id": f"W{i:03d}",
            "worker_name": name,
        })
    return workers


def create_maintenance_logs():
    rows = []
    today = datetime.now().date()

    for zone_index, zone in enumerate(ZONES, start=1):
        zone_has_overdue = random.random() < 0.35
        for machine_no in range(1, 4):
            if zone_has_overdue and machine_no == 1:
                last_date = today - timedelta(days=random.randint(80, 130))
                next_due = today - timedelta(days=random.randint(2, 20))
                status = "Overdue"
            else:
                last_date = today - timedelta(days=random.randint(10, 60))
                if random.random() < 0.25:
                    next_due = today + timedelta(days=random.randint(1, 10))
                    status = "Due Soon"
                else:
                    next_due = today + timedelta(days=random.randint(15, 60))
                    status = "OK"

            rows.append({
                "machine_id": f"M-{zone_index}{machine_no:02d}",
                "zone_id": zone,
                "last_maintenance_date": last_date,
                "next_due_date": next_due,
                "fault_history_count": random.randint(0, 8),
                "maintenance_status": status,
            })

    return pd.DataFrame(rows)


def create_work_permits(workers):
    rows = []
    now = datetime.now()
    task_by_zone = {
        "Zone-A": "Welding inspection",
        "Zone-B": "Packing line check",
        "Zone-C": "Electrical inspection",
        "Furnace Area": "Heat treatment support",
        "Chemical Storage": "Chemical inventory check",
        "Assembly Point": "Safety briefing",
    }

    for worker in workers:
        allowed_zones = random.sample(ZONES, 5)
        for zone in allowed_zones:
            valid_from = now - timedelta(days=random.randint(16, 24))
            valid_to = now + timedelta(days=random.randint(1, 8))
            status = "Valid" if valid_to > now else "Expired"
            if random.random() < 0.12:
                status = random.choice(["Suspended", "Expired"])

            rows.append({
                "permit_id": f"P-{worker['worker_id']}-{zone.replace(' ', '')}",
                "worker_id": worker["worker_id"],
                "zone_id": zone,
                "allowed_task": task_by_zone[zone],
                "valid_from": valid_from,
                "valid_to": valid_to,
                "permit_status": status,
            })

    return pd.DataFrame(rows)


def get_sensor_values(zone):
    base = {
        "Zone-A": (36, 45, 3.5, 5.2, 78),
        "Zone-B": (33, 35, 3.0, 4.8, 75),
        "Zone-C": (34, 40, 4.0, 5.5, 82),
        "Furnace Area": (49, 28, 5.5, 6.0, 88),
        "Chemical Storage": (37, 58, 2.8, 5.0, 76),
        "Assembly Point": (31, 20, 1.5, 3.5, 65),
    }

    temp, gas, vibration, pressure, noise = base[zone]
    return {
        "temperature": round(np.random.normal(temp, 5), 2),
        "gas_level": round(np.random.normal(gas, 12), 2),
        "vibration": round(np.random.normal(vibration, 1.2), 2),
        "pressure": round(np.random.normal(pressure, 1.0), 2),
        "noise_level": round(np.random.normal(noise, 7), 2),
    }


def add_small_missing_values(df, columns, missing_rate=0.015):
    new_df = df.copy()
    for col in columns:
        mask = np.random.rand(len(new_df)) < missing_rate
        new_df.loc[mask, col] = np.nan
    return new_df


def generate_sample_data(row_count=5000):
    make_project_dirs()
    random.seed(42)
    np.random.seed(42)

    workers = create_workers()
    maintenance_df = create_maintenance_logs()
    permits_df = create_work_permits(workers)

    sensor_rows = []
    worker_rows = []
    incident_rows = []

    start_time = datetime.now() - timedelta(minutes=5 * (row_count - 1))
    zone_maintenance = maintenance_df.groupby("zone_id")["maintenance_status"].apply(
        lambda values: "Overdue" if "Overdue" in list(values) else "OK"
    ).to_dict()

    for i in range(row_count):
        timestamp = start_time + timedelta(minutes=5 * i)
        zone = random.choices(
            ZONES,
            weights=[18, 16, 16, 18, 20, 12],
            k=1,
        )[0]
        worker = random.choice(workers)
        values = get_sensor_values(zone)

        if random.random() < 0.08:
            values["gas_level"] += random.uniform(20, 40)
        if random.random() < 0.07:
            values["temperature"] += random.uniform(10, 20)
        if random.random() < 0.08:
            values["vibration"] += random.uniform(2.0, 4.0)
        if random.random() < 0.05:
            values["pressure"] += random.uniform(2.0, 3.0)

        machine_status = "Running"
        if values["vibration"] > 7 or zone_maintenance.get(zone) == "Overdue":
            machine_status = random.choice(["Running", "Warning", "Warning", "Stopped"])

        ppe_status = random.choices(["OK", "Missing Helmet", "Missing Gloves", "No Mask"], [82, 6, 6, 6])[0]
        risk_zone = "High-Risk" if zone in ["Zone-A", "Furnace Area", "Chemical Storage"] else "Normal"
        entry_status = "Restricted" if risk_zone == "High-Risk" and random.random() < 0.18 else "Allowed"

        permit_match = permits_df[
            (permits_df["worker_id"] == worker["worker_id"]) &
            (permits_df["zone_id"] == zone)
        ]
        permit_status = "Invalid"
        if not permit_match.empty:
            permit_status = permit_match.iloc[0]["permit_status"]

        risk_info = calculate_compound_risk({
            **values,
            "zone_id": zone,
            "ppe_status": ppe_status,
            "risk_zone": risk_zone,
            "entry_status": entry_status,
            "permit_status": permit_status,
            "maintenance_status": zone_maintenance.get(zone, "OK"),
        })

        sensor_rows.append({
            "timestamp": timestamp,
            "zone_id": zone,
            "temperature": max(20, round(values["temperature"], 2)),
            "gas_level": max(5, round(values["gas_level"], 2)),
            "vibration": max(0.5, round(values["vibration"], 2)),
            "pressure": max(1, round(values["pressure"], 2)),
            "noise_level": max(45, round(values["noise_level"], 2)),
            "machine_status": machine_status,
            "risk_label": risk_info["risk_level"],
        })

        lat, lon = ZONE_COORDS[zone]
        worker_rows.append({
            "timestamp": timestamp,
            "worker_id": worker["worker_id"],
            "worker_name": worker["worker_name"],
            "zone_id": zone,
            "latitude": round(lat + np.random.normal(0, 0.0007), 6),
            "longitude": round(lon + np.random.normal(0, 0.0007), 6),
            "ppe_status": ppe_status,
            "risk_zone": risk_zone,
            "entry_status": entry_status,
        })

        if risk_info["risk_level"] in ["High", "Critical"] and random.random() < 0.08:
            incident_rows.append({
                "timestamp": timestamp,
                "zone_id": zone,
                "incident_type": random.choice(["Gas alert", "PPE violation", "Machine warning", "Restricted entry"]),
                "severity": risk_info["risk_level"],
                "action_taken": random.choice(["Area checked", "Supervisor informed", "Worker moved", "Machine inspected"]),
            })

    sensor_df = pd.DataFrame(sensor_rows)
    worker_df = pd.DataFrame(worker_rows)
    incident_df = pd.DataFrame(incident_rows)

    sensor_df = add_small_missing_values(sensor_df, ["temperature", "gas_level", "vibration", "pressure", "noise_level"])
    worker_df = add_small_missing_values(worker_df, ["ppe_status", "entry_status"])

    sensor_df.to_csv(SAMPLE_DIR / "sensor_data.csv", index=False)
    worker_df.to_csv(SAMPLE_DIR / "worker_location.csv", index=False)
    maintenance_df.to_csv(SAMPLE_DIR / "maintenance_logs.csv", index=False)
    permits_df.to_csv(SAMPLE_DIR / "work_permits.csv", index=False)
    incident_df.to_csv(SAMPLE_DIR / "incident_logs.csv", index=False)

    return {
        "sensor_rows": len(sensor_df),
        "worker_rows": len(worker_df),
        "maintenance_rows": len(maintenance_df),
        "permit_rows": len(permits_df),
        "incident_rows": len(incident_df),
    }


if __name__ == "__main__":
    print(generate_sample_data())
