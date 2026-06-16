from .correctness import verify_correctness
from .completeness import verify_completeness
from .consistency import verify_consistency


def run_verification(adl: dict):

    style = adl.get("style")
    if not style:
        return {
            "status": "NOT_VERIFIED",
            "failed_layer": "consistency",
            "details": {
                "layer": "consistency",
                "status": "FAILED",
                "issues": [
                    {
                        "rule": "MISSING_STYLE",
                        "message": "Architecture style is required."
                    }
                ]
            }
        }
    
    correctness = verify_correctness(adl)
    completeness = verify_completeness(adl)
    consistency = verify_consistency(adl)

    # Fold the missing-style issue into the consistency layer so the
    # downstream report can render it consistently.
    
    layers = {
        "correctness": correctness,
        "completeness": completeness,
        "consistency": consistency,
    }

    failed_layers = [name for name, result in layers.items()
                     if result.get("status") == "FAILED"]

    return {
        "status": "VERIFIED" if not failed_layers else "NOT_VERIFIED",
        "failed_layers": failed_layers,
        "layers": layers,
    }