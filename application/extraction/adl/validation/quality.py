# application/extraction/adl/validation/quality.py

def validate_quality(adl: dict):
    components = adl.get("components", [])
    relationships = adl.get("relationships", [])

    names = {c["name"] for c in components}

    fan_in = {n: 0 for n in names}
    fan_out = {n: 0 for n in names}

    for r in relationships:
        src = r.get("source")
        tgt = r.get("target")

        if src in fan_out:
            fan_out[src] += 1

        if tgt in fan_in:
            fan_in[tgt] += 1

    risks = []

    for n in names:
        coupling = fan_in[n] + fan_out[n]
        if coupling > 4:
            risks.append({
                "component": n,
                "risk": "High coupling",
                "quality_attribute": "Modifiability"
            })

    centralization_index = max(fan_in.values()) / len(names) if names else 0

    if centralization_index > 0.6:
        risks.append({
            "risk": "High centralization",
            "quality_attribute": "Availability"
        })

    return {
        "fan_in": fan_in,
        "fan_out": fan_out,
        "centralization_index": centralization_index,
        "risks": risks
    }