from adl.ai_engine import ai_generate_architecture
from adl.architecture_metrics import compute_metrics
from adl.architecture_validation import validate_architecture


def generate_adl(req):
    # -------- AI Architecture Generation --------
    ai = ai_generate_architecture(
        req["system_name"],
        req["functional_requirements"],
        req["non_functional_requirements"],
        req["architecture_style"]
    )

    # -------- Level 2.1: Metrics --------
    metrics = compute_metrics({
        "services": ai["components"],
        "relationships": ai["relationships"]
    })

    # -------- Level 2.2: Validation --------
    validation = validate_architecture({
        "metrics": metrics,
        "qualityAttributes": req["non_functional_requirements"]
    })

    adl = {
        "system": {
            "name": req["system_name"],
            "architectureStyle": req["architecture_style"]
        },

        "qualityAttributes": req["non_functional_requirements"],

        "decisions": ai["decisions"],

        "views": {
            "logical": {
                "services": ai["components"],
                "relationships": ai["relationships"]
            },
            "deployment": ai["production_intents"]["deployment"],
            "security": ai["production_intents"]["security"],
            "resilience": ai["production_intents"]["resilience"]
        },

        "metrics": metrics,

        # Backward compatibility
        "services": ai["components"],
        "relationships": ai["relationships"],

        "critique": ai["critique"]
    }

    return adl, validation
