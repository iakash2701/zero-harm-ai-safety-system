from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.risk_engine import calculate_compound_risk
from src.utils import PROCESSED_DIR, REPORT_DIR, make_project_dirs


def make_alert_rows(df, limit=25):
    alerts = []
    recent_rows = df.sort_values("timestamp", ascending=False).head(300)

    for _, row in recent_rows.iterrows():
        result = calculate_compound_risk(row.to_dict())
        if result["risk_level"] in ["High", "Critical"]:
            alerts.append({
                "timestamp": row["timestamp"],
                "zone_id": row["zone_id"],
                "worker_name": row["worker_name"],
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"],
                "reasons": "; ".join(result["detected_reasons"][:3]),
                "recommended_actions": "; ".join(result["recommended_actions"][:3]),
            })

    return pd.DataFrame(alerts).head(limit)


def create_summary(df):
    return {
        "total_records": len(df),
        "total_alerts": int(df["risk_label"].isin(["High", "Critical"]).sum()),
        "critical_risks": int((df["risk_label"] == "Critical").sum()),
        "high_risk_zones": ", ".join(df[df["risk_label"].isin(["High", "Critical"])]["zone_id"].value_counts().head(3).index),
        "worker_violations": int((df["ppe_status"] != "OK").sum()),
        "maintenance_issues": int((df["maintenance_status"] == "Overdue").sum()),
        "permit_violations": int((df["permit_status"] != "Valid").sum()),
        "average_gas_level": round(df["gas_level"].mean(), 2),
        "average_temperature": round(df["temperature"].mean(), 2),
        "average_vibration": round(df["vibration"].mean(), 2),
    }


def save_pdf_report(summary, alerts_df):
    pdf_path = REPORT_DIR / "daily_safety_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Daily Industrial Safety Report", styles["Title"]))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    summary_data = [["Metric", "Value"]]
    for key, value in summary.items():
        summary_data.append([key.replace("_", " ").title(), str(value)])

    table = Table(summary_data, colWidths=[190, 300])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Latest High Priority Alerts", styles["Heading2"]))
    if alerts_df.empty:
        story.append(Paragraph("No high priority alerts found in the latest records.", styles["Normal"]))
    else:
        alert_data = [["Time", "Zone", "Worker", "Risk", "Reason"]]
        for _, row in alerts_df.head(10).iterrows():
            alert_data.append([
                str(row["timestamp"])[:16],
                row["zone_id"],
                row["worker_name"],
                f"{row['risk_level']} ({row['risk_score']})",
                row["reasons"][:90],
            ])

        alert_table = Table(alert_data, colWidths=[75, 85, 85, 70, 185])
        alert_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(alert_table)

    doc.build(story)
    return pdf_path


def generate_reports():
    make_project_dirs()
    df = pd.read_csv(PROCESSED_DIR / "final_safety_dataset.csv")
    summary = create_summary(df)
    alerts_df = make_alert_rows(df)

    pd.DataFrame([summary]).to_csv(REPORT_DIR / "daily_summary.csv", index=False)
    alerts_df.to_csv(REPORT_DIR / "daily_safety_alerts.csv", index=False)
    pdf_path = save_pdf_report(summary, alerts_df)

    return {
        "summary": REPORT_DIR / "daily_summary.csv",
        "alerts": REPORT_DIR / "daily_safety_alerts.csv",
        "pdf": pdf_path,
    }


if __name__ == "__main__":
    print(generate_reports())
