# application/extraction/adl/validation/runner.py

from .structural import validate_structural
from .quality import validate_quality
from .domain import validate_domain


def run_validation(adl: dict):
    structural = validate_structural(adl)
    quality = validate_quality(adl)
    domain = validate_domain(adl)

    # --------------------------------------------------
    # ORIGINAL LOGIC (UNCHANGED)
    # Structural validity is the primary failure condition
    # --------------------------------------------------
    status = "FAILED" if structural["failed"] else "SUCCESS"

    # --------------------------------------------------
    # FAILURE POLICIES (ADDITIVE – does NOT remove old logic)
    # --------------------------------------------------

    # Policy 1: Accumulation of quality risks (ATAM)
    quality_risks = len(quality.get("risks", []))
    if quality_risks >= 2:
        status = "FAILED"

    # Policy 2: Domain unsuitability for Microservices
    # (Synchronous communication + high state concentration)
    if (
        adl.get("style") == "Microservices"
        and domain.get("communication_style") == "synchronous"
        and domain.get("state_concentration", 0) > 3
    ):
        status = "FAILED"

    # Policy 3: Combined quality + domain degradation
    if quality_risks >= 1 and domain.get("state_concentration", 0) > 3:
        status = "FAILED"

    # --------------------------------------------------
    # FINAL RESULT (same structure as before)
    # --------------------------------------------------
    return {
        "status": status,
        "structural": structural,
        "quality": quality,
        "domain": domain
    }