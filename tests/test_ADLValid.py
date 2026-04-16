# tests/test_structural_validation.py

import pytest
import os
from application.extraction.adl.validation.structural import validate_structural
from application.extraction.adl.validation.quality import validate_quality
from application.extraction.adl.validation.domain import validate_domain
from application.extraction.adl.verification.runner import run_verification
from application.extraction.adl.verification.verification_report_generator import generate_verification_pdf



def test_tc_f4_structural_validation():

    adl = {
        "components": [
            {"name": "A"},
            {"name": "B"}
        ],
        "relationships": [
            {"source": "A", "target": "B", "type": "event-flow"},
            {"source": "A", "target": "C", "type": "data-flow"},  # ❌ undefined
            {"source": "B", "target": "B", "type": "event-flow"}, # ❌ self loop
            {"source": "A", "target": "B", "type": "wrong"}       # ❌ invalid type
        ]
    }

    result = validate_structural(adl)

    # structure check
    assert result["failed"] is True

    metrics = result["metrics"]

    assert metrics["undefined_references"] == 1
    assert metrics["self_loops"] == 1
    assert metrics["invalid_relationship_types"] == 1

    # validity ratio
    assert result["validity_ratio"] < 1

##################

def test_tc_f5_high_coupling_detection():

    adl1 = {
        "components": [
            {"name": "A"},
            {"name": "B"},
            {"name": "C"},
            {"name": "D"}
        ],
        "relationships": [
            {"source": "A", "target": "B"},
            {"source": "A", "target": "C"},
            {"source": "A", "target": "D"},
            {"source": "B", "target": "A"},
            {"source": "C", "target": "A"}  # 🔥 يخلي coupling > 4
        ]
    }

    result = validate_quality(adl1)

    # =========================
    # 🔥 ASSERTIONS
    # =========================

    assert "risks" in result
    assert len(result["risks"]) > 0

    # check high coupling detected
    risks = result["risks"]

    assert any(r["risk"] == "High coupling" for r in risks)

##################

def test_tc_f6_communication_style():

    adl2 = {
        "components": [
            {"name": "A"},
            {"name": "B"}
        ],
        "relationships": [
            {"source": "A", "target": "B", "type": "event-flow"},
            {"source": "B", "target": "A", "type": "event-flow"}
        ]
    }

    result = validate_domain(adl2)

    # =========================
    # 🔥 STRONG ASSERTIONS
    # =========================

    # structure check
    assert isinstance(result, dict)

    # keys existence
    assert "communication_style" in result
    assert "stateless_ratio" in result
    assert "state_concentration" in result

    # value correctness
    assert result["communication_style"] == "event-driven"

    # additional checks (based on logic)
    assert result["stateless_ratio"] == 1.0
    assert result["state_concentration"] >= 1

    ##################




import os

def test_tc_f7_generate_verification_report():

    adl = {
        "style": "Microservices",
        "components": [
            {"name": "A"},
            {"name": "B"}
        ],
        "relationships": [
            {"source": "A", "target": "B", "type": "event-flow"}
        ]
    }

    # 1️⃣ run verification
    verification_result = run_verification(adl)

    assert verification_result["status"] == "VERIFIED"

    # 2️⃣ generate report
    report_path = generate_verification_pdf(verification_result)

    # =========================
    # 🔥 STRONG ASSERTIONS
    # =========================

    assert report_path is not None
    assert isinstance(report_path, str)
    assert report_path.endswith(".pdf")

    # 🔥 important
    assert os.path.exists(report_path)