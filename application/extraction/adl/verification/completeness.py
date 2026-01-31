def verify_completeness(adl: dict):
    issues = []

    components = adl.get("components", [])
    relationships = adl.get("relationships", [])

    component_names = {c.get("name") for c in components}

    # -------- Rule 1: Relationships existence --------
    if not relationships:
        issues.append({
            "rule": "NO_RELATIONSHIPS",
            "message": "Architecture contains components with no interactions."
        })

    # -------- Rule 2: Isolated components --------
    for name in component_names:
        involved = any(
            r.get("source") == name or r.get("target") == name
            for r in relationships
        )
        if not involved:
            issues.append({
                "rule": "ISOLATED_COMPONENT",
                "message": f"Component '{name}' has no interactions."
            })

    # -------- Rule 3: Architecture intent --------
    style = adl.get("style")
    if style and style.lower() == "microservices":
        if len(component_names) < 2:
            issues.append({
                "rule": "INSUFFICIENT_COMPONENTS",
                "message": "Microservices architecture must contain multiple services."
            })

    return {
        "layer": "completeness",
        "status": "FAILED" if issues else "PASSED",
        "issues": issues
    }