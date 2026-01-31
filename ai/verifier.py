# adl/verifier.py

def verify_completeness(adl):
    if not adl.components:
        raise ValueError("ADL must contain at least one component")
    if not adl.connectors:
        raise ValueError("ADL must contain at least one connector")

def verify_consistency(adl):
    component_names = {c.name for c in adl.components}

    for conn in adl.connectors:
        if conn.source not in component_names:
            raise ValueError(f"Invalid connector source: {conn.source}")
        if conn.target not in component_names:
            raise ValueError(f"Invalid connector target: {conn.target}")