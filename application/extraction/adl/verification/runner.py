from .correctness import verify_correctness
from .completeness import verify_completeness
from .consistency import verify_consistency

def run_verification(adl: dict):
    correctness = verify_correctness(adl)
    if correctness["status"] == "FAILED":
        return {
            "status": "NOT_VERIFIED",
            "failed_layer": "correctness",
            "details": correctness
        }

    completeness = verify_completeness(adl)
    if completeness["status"] == "FAILED":
        return {
            "status": "NOT_VERIFIED",
            "failed_layer": "completeness",
            "details": completeness
        }

    consistency = verify_consistency(adl)
    if consistency["status"] == "FAILED":
        return {
            "status": "NOT_VERIFIED",
            "failed_layer": "consistency",
            "details": consistency
        }

    return {
        "status": "VERIFIED"
    }