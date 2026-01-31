from application.extraction.adl.validation.quality import validate_quality
from application.extraction.adl.validation.domain import validate_domain

def verify_consistency(adl: dict):
    issues = []

    quality = validate_quality(adl)
    domain = validate_domain(adl)

    fan_in = quality.get("fan_in", {})
    fan_out = quality.get("fan_out", {})

    # -------- Rule 1: Metric consistency --------
    for comp in fan_in:
        if fan_in[comp] < 0 or fan_out[comp] < 0:
            issues.append({
                "rule": "NEGATIVE_METRIC",
                "message": f"Negative fan-in or fan-out detected for '{comp}'."
            })

    # -------- Rule 2: Centralization bounds --------
    centralization_index = quality.get("centralization_index", 0)
    if not (0 <= centralization_index <= 1):
        issues.append({
            "rule": "CENTRALIZATION_OUT_OF_RANGE",
            "message": "Centralization index is outside [0,1]."
        })

    # -------- Rule 3: Style vs Domain consistency --------
    if adl.get("style") == "Microservices":
        if domain.get("communication_style") == "synchronous":
            issues.append({
                "rule": "STYLE_COMMUNICATION_CONFLICT",
                "message": "Microservices architecture uses synchronous communication."
            })

        if domain.get("state_concentration", 0) > 3:
            issues.append({
                "rule": "STYLE_STATE_CONFLICT",
                "message": "High state concentration conflicts with microservices style."
            })

    return {
        "layer": "consistency",
        "status": "FAILED" if issues else "PASSED",
        "issues": issues
    }