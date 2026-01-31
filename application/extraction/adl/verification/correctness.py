def verify_correctness(adl: dict):
    violations = []

    components = adl.get("components", [])
    relationships = adl.get("relationships", [])

    # -------- Rule 1: Components existence --------
    if not components:
        violations.append({
            "rule": "COMPONENT_EXISTENCE",
            "message": "No components defined in the architecture."
        })

    component_names = set()

    # -------- Rule 2: Unique component names --------
    for c in components:
        name = c.get("name")
        if not name:
            violations.append({
                "rule": "COMPONENT_NAMING",
                "message": "A component is missing a name."
            })
        elif name in component_names:
            violations.append({
                "rule": "COMPONENT_UNIQUENESS",
                "message": f"Duplicate component name detected: '{name}'."
            })
        component_names.add(name)

    # -------- Rule 3: Relationship correctness --------
    allowed_types = {"event-flow", "data-flow"}

    for r in relationships:
        src = r.get("source")
        tgt = r.get("target")
        rtype = r.get("type")

        if src not in component_names:
            violations.append({
                "rule": "UNDEFINED_SOURCE",
                "message": f"Relationship source '{src}' is not a defined component."
            })

        if tgt not in component_names:
            violations.append({
                "rule": "UNDEFINED_TARGET",
                "message": f"Relationship target '{tgt}' is not a defined component."
            })

        if src == tgt:
            violations.append({
                "rule": "SELF_LOOP",
                "message": f"Self-loop detected on component '{src}'."
            })

        if rtype not in allowed_types:
            violations.append({
                "rule": "INVALID_RELATIONSHIP_TYPE",
                "message": f"Invalid relationship type '{rtype}'."
            })

    return {
        "layer": "correctness",
        "status": "FAILED" if violations else "PASSED",
        "violations": violations
    }