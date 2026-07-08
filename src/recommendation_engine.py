def get_recommendations(reasons):
    actions = []

    text = " ".join(reasons).lower()

    if "gas" in text:
        actions.extend([
            "Evacuate workers from the affected zone.",
            "Start ventilation and stop ignition sources.",
            "Notify the safety officer immediately.",
        ])

    if "temperature" in text or "heat" in text:
        actions.extend([
            "Reduce worker exposure time in the hot area.",
            "Check cooling system and heat shields.",
        ])

    if "vibration" in text or "maintenance" in text:
        actions.extend([
            "Shut down the machine if vibration keeps increasing.",
            "Schedule urgent maintenance inspection.",
        ])

    if "ppe" in text:
        actions.extend([
            "Alert the supervisor about PPE violation.",
            "Restrict worker entry until PPE is corrected.",
        ])

    if "permit" in text or "restricted" in text:
        actions.extend([
            "Verify work permit before allowing entry.",
            "Record the unauthorized entry event.",
        ])

    if not actions:
        actions.append("Continue monitoring and follow standard safety checklist.")

    return list(dict.fromkeys(actions))
