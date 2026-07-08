from src.recommendation_engine import get_recommendations
from src.utils import get_risk_level


def calculate_compound_risk(record):
    score = 10
    reasons = []

    gas = float(record.get("gas_level", 0))
    temperature = float(record.get("temperature", 0))
    vibration = float(record.get("vibration", 0))
    pressure = float(record.get("pressure", 0))
    noise = float(record.get("noise_level", 0))
    ppe_status = str(record.get("ppe_status", "OK"))
    risk_zone = str(record.get("risk_zone", "Normal"))
    entry_status = str(record.get("entry_status", "Allowed"))
    permit_status = str(record.get("permit_status", "Valid"))
    maintenance_status = str(record.get("maintenance_status", "OK"))
    zone_id = str(record.get("zone_id", "Unknown Zone"))

    if gas > 70:
        score += 25
        reasons.append("High gas level detected")
    elif gas > 55:
        score += 12
        reasons.append("Gas level is above normal")

    if temperature > 55:
        score += 18
        reasons.append("High temperature in work area")
    elif temperature > 45:
        score += 9
        reasons.append("Temperature is slightly high")

    if vibration > 7:
        score += 18
        reasons.append("High machine vibration")
    elif vibration > 5.5:
        score += 8
        reasons.append("Machine vibration needs attention")

    if pressure > 8:
        score += 15
        reasons.append("Pressure level is high")

    if noise > 90:
        score += 10
        reasons.append("Noise level may affect worker safety")

    if maintenance_status == "Overdue":
        score += 15
        reasons.append("Maintenance is overdue")

    if ppe_status != "OK":
        score += 18
        reasons.append("PPE violation found")

    if risk_zone == "High-Risk":
        score += 10
        reasons.append(f"Worker present in high-risk zone {zone_id}")

    if entry_status == "Restricted":
        score += 12
        reasons.append("Worker entered restricted zone")

    if permit_status != "Valid":
        score += 14
        reasons.append("Invalid or expired work permit")

    if gas > 70 and zone_id in ["Zone-A", "Chemical Storage"]:
        score += 20
        reasons.append("Critical gas exposure risk")

    if vibration > 7 and maintenance_status == "Overdue":
        score += 18
        reasons.append("Machine failure risk due to vibration and overdue maintenance")

    if temperature > 50 and ppe_status != "OK":
        score += 15
        reasons.append("Heat safety violation because PPE is missing")

    if entry_status == "Restricted" and permit_status != "Valid":
        score += 18
        reasons.append("Unauthorized hazardous zone entry")

    medium_issue_count = sum([
        gas > 55,
        temperature > 45,
        vibration > 5.5,
        pressure > 7,
        noise > 85,
        ppe_status != "OK",
        maintenance_status == "Overdue",
        permit_status != "Valid",
    ])
    if medium_issue_count >= 4:
        score += 15
        reasons.append("Multiple medium risks are happening together")

    score = min(score, 100)
    risk_level = get_risk_level(score)

    if not reasons:
        reasons.append("No major issue detected")

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "detected_reasons": reasons,
        "recommended_actions": get_recommendations(reasons),
    }
