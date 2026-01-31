# application/extraction/adl/validation/structural.py

def validate_structural(adl: dict):
    components = adl.get("components", [])
    relationships = adl.get("relationships", [])

    component_names = {c.get("name") for c in components}

    metrics = {
        "component_count": len(components),
        "relationship_count": len(relationships),
        "undefined_references": 0,
        "self_loops": 0,
        "invalid_relationship_types": 0
    }

    for rel in relationships:
        src = rel.get("source")
        tgt = rel.get("target")
        rtype = rel.get("type")

        if src not in component_names or tgt not in component_names:
            metrics["undefined_references"] += 1

        if src == tgt:
            metrics["self_loops"] += 1

        if rtype not in {"event-flow", "data-flow"}:
            metrics["invalid_relationship_types"] += 1

    total_issues = (
        metrics["undefined_references"] +
        metrics["self_loops"] +
        metrics["invalid_relationship_types"]
    )

    validity_ratio = (
        (metrics["relationship_count"] - total_issues) /
        metrics["relationship_count"]
        if metrics["relationship_count"] > 0 else 0
    )

    return {
        "metrics": metrics,
        "validity_ratio": validity_ratio,
        "failed": total_issues > 0
    }