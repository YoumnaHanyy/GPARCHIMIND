
# application/extraction/adl/validation/domain.py

def validate_domain(adl: dict):
    components = adl.get("components", [])
    relationships = adl.get("relationships", [])

    total = len(components)

    # Stateless ratio (inferred)
    stateless_ratio = 1.0  # because no state declared per component

    fan_in = {}
    for r in relationships:
        fan_in[r["target"]] = fan_in.get(r["target"], 0) + 1

    state_concentration = max(fan_in.values()) if fan_in else 0

    event_flows = sum(1 for r in relationships if r["type"] == "event-flow")
    data_flows = sum(1 for r in relationships if r["type"] == "data-flow")

    communication_style = (
        "event-driven" if event_flows >= data_flows else "synchronous"
    )

    return {
        "stateless_ratio": stateless_ratio,
        "state_concentration": state_concentration,
        "communication_style": communication_style
    }