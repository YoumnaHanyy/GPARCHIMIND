def validate_architecture(adl):
    metrics = adl.get("metrics", {})
    qa = adl.get("qualityAttributes", {})

    warnings = []
    errors = []

    # ---------- Rule 1: Latency vs Critical Path ----------
    if qa.get("latency") in ["Very Low", "Low"]:
        if metrics.get("critical_path_length", 0) > 4:
            errors.append(
                "Critical path is too long for low-latency requirement"
            )

    # ---------- Rule 2: God Component ----------
    if metrics.get("max_fan_out", 0) >= 6:
        warnings.append(
            "Possible God component detected (high fan-out)"
        )

    # ---------- Rule 3: Over-synchronous Architecture ----------
    if metrics.get("async_ratio", 1) < 0.3:
        warnings.append(
            "Low async ratio may reduce scalability"
        )

    # ---------- Rule 4: Over-engineering ----------
    if metrics.get("num_components", 0) > 15:
        warnings.append(
            "High number of components may indicate over-engineering"
        )

    # ---------- Rule 5: Under-decomposition ----------
    if metrics.get("num_components", 0) < 3:
        errors.append(
            "Too few components for a production system"
        )

    return {
        "errors": errors,
        "warnings": warnings,
        "is_valid": len(errors) == 0
    }
