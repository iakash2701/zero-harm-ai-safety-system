import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.risk_engine import calculate_compound_risk
from src.train_model import FEATURE_COLUMNS
from src.utils import MODEL_DIR, PROCESSED_DIR, REPORT_DIR, SAMPLE_DIR


st.set_page_config(
    page_title="Industrial Safety AI",
    page_icon="⚠️",
    layout="wide",
)


def load_data():
    data_path = PROCESSED_DIR / "final_safety_dataset.csv"
    if not data_path.exists():
        st.warning("Dataset not found. Please run `python run_project.py` first.")
        st.stop()
    return pd.read_csv(data_path, parse_dates=["timestamp"])


def load_model():
    model_path = MODEL_DIR / "risk_prediction_model.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def show_kpis(df):
    active_alerts = df[df["risk_label"].isin(["High", "Critical"])]
    overdue = df[df["maintenance_status"] == "Overdue"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Zones", df["zone_id"].nunique())
    c2.metric("Total Workers", df["worker_id"].nunique())
    c3.metric("Active Alerts", len(active_alerts))
    c4.metric("Critical Risks", int((df["risk_label"] == "Critical").sum()))
    c5.metric("Maintenance Overdue", overdue["zone_id"].nunique())


def latest_alerts(df):
    alerts = df[df["risk_label"].isin(["High", "Critical"])].sort_values("timestamp", ascending=False)
    cols = ["timestamp", "zone_id", "worker_name", "risk_label", "gas_level", "temperature", "ppe_status", "permit_status"]
    return alerts[cols].head(15)


def overview_page(df):
    st.subheader("Overview")
    show_kpis(df)

    left, right = st.columns(2)
    with left:
        risk_count = df["risk_label"].value_counts().reset_index()
        risk_count.columns = ["Risk Level", "Count"]
        st.plotly_chart(px.bar(risk_count, x="Risk Level", y="Count", color="Risk Level"), use_container_width=True)

    with right:
        zone_count = df[df["risk_label"].isin(["High", "Critical"])]["zone_id"].value_counts().reset_index()
        zone_count.columns = ["Zone", "Alert Count"]
        st.plotly_chart(px.bar(zone_count, x="Zone", y="Alert Count", color="Zone"), use_container_width=True)

    st.subheader("Latest Safety Alerts")
    st.dataframe(latest_alerts(df), use_container_width=True)


def sensor_page(df):
    st.subheader("Live Sensor Monitoring")
    zone = st.selectbox("Select Zone", sorted(df["zone_id"].unique()))
    zone_df = df[df["zone_id"] == zone].sort_values("timestamp").tail(120)

    st.plotly_chart(px.line(zone_df, x="timestamp", y="gas_level", title="Gas Level Trend"), use_container_width=True)
    st.plotly_chart(px.line(zone_df, x="timestamp", y="temperature", title="Temperature Trend"), use_container_width=True)
    st.plotly_chart(px.line(zone_df, x="timestamp", y="vibration", title="Vibration Trend"), use_container_width=True)

    st.write("Recent sensor records")
    st.dataframe(zone_df[["timestamp", "zone_id", "temperature", "gas_level", "vibration", "pressure", "noise_level", "risk_label"]].tail(20), use_container_width=True)


def prediction_page(df):
    st.subheader("AI Risk Prediction")
    model_bundle = load_model()
    if model_bundle is None:
        st.warning("Model file not found. Please run `python run_project.py` first.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        zone = st.selectbox("Zone", sorted(df["zone_id"].unique()))
        temperature = st.number_input("Temperature", 20.0, 90.0, 42.0)
        gas_level = st.number_input("Gas Level", 0.0, 150.0, 50.0)
        vibration = st.number_input("Vibration", 0.0, 15.0, 4.5)
    with c2:
        pressure = st.number_input("Pressure", 1.0, 15.0, 5.5)
        noise_level = st.number_input("Noise Level", 40.0, 120.0, 80.0)
        machine_status = st.selectbox("Machine Status", ["Running", "Warning", "Stopped"])
        fault_history_count = st.number_input("Fault History Count", 0.0, 15.0, 2.0)
    with c3:
        ppe_status = st.selectbox("PPE Status", ["OK", "Missing Helmet", "Missing Gloves", "No Mask"])
        risk_zone = st.selectbox("Risk Zone", ["Normal", "High-Risk"])
        entry_status = st.selectbox("Entry Status", ["Allowed", "Restricted"])
        maintenance_status = st.selectbox("Maintenance Status", ["OK", "Due Soon", "Overdue"])
        permit_status = st.selectbox("Permit Status", ["Valid", "Expired", "Suspended", "Invalid"])

    input_data = {
        "zone_id": zone,
        "temperature": temperature,
        "gas_level": gas_level,
        "vibration": vibration,
        "pressure": pressure,
        "noise_level": noise_level,
        "machine_status": machine_status,
        "ppe_status": ppe_status,
        "risk_zone": risk_zone,
        "entry_status": entry_status,
        "fault_history_count": fault_history_count,
        "maintenance_status": maintenance_status,
        "permit_status": permit_status,
        "maintenance_overdue": int(maintenance_status == "Overdue"),
        "ppe_violation": int(ppe_status != "OK"),
        "invalid_permit": int(permit_status != "Valid"),
        "restricted_entry": int(entry_status == "Restricted"),
    }

    if st.button("Predict Risk"):
        row = pd.DataFrame([input_data])[FEATURE_COLUMNS]
        prediction = model_bundle["model"].predict(row)[0]
        label = model_bundle["label_encoder"].inverse_transform([prediction])[0]
        confidence = max(model_bundle["model"].predict_proba(row)[0])
        rule_result = calculate_compound_risk(input_data)

        st.success(f"Predicted Risk Level: {label}")
        st.info(f"Model confidence: {confidence:.2f}")
        st.metric("Rule-Based Risk Score", rule_result["risk_score"])

        st.write("Detected Reasons")
        for reason in rule_result["detected_reasons"]:
            st.write(f"- {reason}")

        st.write("Recommended Preventive Actions")
        for action in rule_result["recommended_actions"]:
            st.write(f"- {action}")

        st.caption("Risk score is calculated from practical safety rules. The ML model separately predicts the risk class from past synthetic records.")


def compound_alerts_page(df):
    st.subheader("Compound Risk Alerts")
    records = []
    for _, row in df.sort_values("timestamp", ascending=False).head(200).iterrows():
        result = calculate_compound_risk(row.to_dict())
        if result["risk_level"] in ["High", "Critical"]:
            records.append({
                "timestamp": row["timestamp"],
                "zone_id": row["zone_id"],
                "worker_name": row["worker_name"],
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"],
                "reasons": "; ".join(result["detected_reasons"][:3]),
                "actions": "; ".join(result["recommended_actions"][:3]),
            })
    st.dataframe(pd.DataFrame(records).head(40), use_container_width=True)


def worker_map_page(df):
    st.subheader("Worker Location Map")
    latest = df.sort_values("timestamp").groupby("worker_id").tail(1)
    st.map(latest[["latitude", "longitude"]])
    st.dataframe(latest[["worker_id", "worker_name", "zone_id", "ppe_status", "risk_zone", "entry_status"]], use_container_width=True)


def maintenance_page():
    st.subheader("Maintenance Status")
    maintenance_path = SAMPLE_DIR / "maintenance_logs.csv"
    if maintenance_path.exists():
        maintenance = pd.read_csv(maintenance_path)
        st.dataframe(maintenance, use_container_width=True)
        status_count = maintenance["maintenance_status"].value_counts().reset_index()
        status_count.columns = ["Status", "Count"]
        st.plotly_chart(px.pie(status_count, names="Status", values="Count"), use_container_width=True)


def permit_page():
    st.subheader("Work Permit Status")
    permit_path = SAMPLE_DIR / "work_permits.csv"
    if permit_path.exists():
        permits = pd.read_csv(permit_path)
        st.dataframe(permits, use_container_width=True)
        status_count = permits["permit_status"].value_counts().reset_index()
        status_count.columns = ["Status", "Count"]
        st.plotly_chart(px.bar(status_count, x="Status", y="Count", color="Status"), use_container_width=True)


def recommendations_page(df):
    st.subheader("Safety Recommendations")
    sample = latest_alerts(df).head(8)
    for _, row in sample.iterrows():
        full_row = df[(df["timestamp"] == row["timestamp"]) & (df["zone_id"] == row["zone_id"])].iloc[0]
        result = calculate_compound_risk(full_row.to_dict())
        with st.expander(f"{row['zone_id']} - {row['risk_label']} - {row['worker_name']}"):
            st.write("Reasons")
            for reason in result["detected_reasons"]:
                st.write(f"- {reason}")
            st.write("Recommended Actions")
            for action in result["recommended_actions"]:
                st.write(f"- {action}")


def reports_page():
    st.subheader("Reports")
    report_files = [
        REPORT_DIR / "daily_summary.csv",
        REPORT_DIR / "daily_safety_alerts.csv",
        REPORT_DIR / "model_comparison.csv",
        REPORT_DIR / "anomaly_report.csv",
        REPORT_DIR / "daily_safety_report.pdf",
    ]

    for file in report_files:
        if file.exists():
            st.write(file.name)
            if file.suffix == ".csv":
                st.dataframe(pd.read_csv(file).head(30), use_container_width=True)
            else:
                st.info(f"PDF report saved at: {file}")


def main():
    st.title("AI-Powered Industrial Safety Intelligence")
    st.caption("Hackathon prototype for zero-harm industrial operations")

    df = load_data()

    page = st.sidebar.radio(
        "Dashboard Pages",
        [
            "Overview",
            "Live Sensor Monitoring",
            "AI Risk Prediction",
            "Compound Risk Alerts",
            "Worker Location Map",
            "Maintenance Status",
            "Work Permit Status",
            "Safety Recommendations",
            "Reports",
        ],
    )

    if page == "Overview":
        overview_page(df)
    elif page == "Live Sensor Monitoring":
        sensor_page(df)
    elif page == "AI Risk Prediction":
        prediction_page(df)
    elif page == "Compound Risk Alerts":
        compound_alerts_page(df)
    elif page == "Worker Location Map":
        worker_map_page(df)
    elif page == "Maintenance Status":
        maintenance_page()
    elif page == "Work Permit Status":
        permit_page()
    elif page == "Safety Recommendations":
        recommendations_page(df)
    elif page == "Reports":
        reports_page()


if __name__ == "__main__":
    main()
