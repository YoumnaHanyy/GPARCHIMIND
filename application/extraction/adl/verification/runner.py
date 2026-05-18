from .correctness import verify_correctness
from .completeness import verify_completeness
from .consistency import verify_consistency


def run_verification(adl: dict):
    """
    Run the three verification layers (correctness, completeness, consistency)
    on the supplied ADL and return an aggregated, per-layer result.

    Unlike a short-circuit gate, this runner ALWAYS executes every layer so
    the verification report can render the complete picture (which rules
    were checked, which passed, which violations were observed).

    Return shape:
    {
        "status": "VERIFIED" | "NOT_VERIFIED",
        "failed_layers": [<layer names that FAILED>],
        "layers": {
            "correctness":  {"layer": "correctness",  "status": "PASSED"|"FAILED", "violations": [...]},
            "completeness": {"layer": "completeness", "status": "PASSED"|"FAILED", "issues": [...]},
            "consistency":  {"layer": "consistency",  "status": "PASSED"|"FAILED", "issues": [...]}
        }
    }

    A missing architectural style is reported as a consistency-layer issue
    rather than aborting verification, so the report still describes the
    other layers.
    """

    # -------- Pre-check: missing style is treated as a consistency issue --------
    style_issues = []
    if not adl.get("style"):
        style_issues.append({
            "rule": "MISSING_STYLE",
            "message": "Architecture style is required."
        })

    # -------- Run all three layers unconditionally --------
    correctness = verify_correctness(adl)
    completeness = verify_completeness(adl)
    consistency = verify_consistency(adl)

    # Fold the missing-style issue into the consistency layer so the
    # downstream report can render it consistently.
    if style_issues:
        consistency = {
            "layer": "consistency",
            "status": "FAILED",
            "issues": style_issues + consistency.get("issues", [])
        }

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