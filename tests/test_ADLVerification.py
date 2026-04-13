import os
import pytest

os.environ["HF_API_KEY"] = "fake_key"

from application.extraction.adl.verification.runner import run_verification
from application.extraction.adl.verification.correctness import verify_correctness
from application.extraction.adl.verification.completeness import verify_completeness
from application.extraction.adl.verification.consistency import verify_consistency

# =========================================
# 🛠️ Data Fixtures
# =========================================

@pytest.fixture
def perfect_adl():
    return {
        "style": "Microservices",
        "components": [
            {"name": "Gateway"},
            {"name": "OrderService"}
        ],
        "relationships": [
            {"source": "Gateway", "target": "OrderService", "type": "event-flow"}
        ]
    }

# =========================================
# ✅ Unit Tests: نختبر كل Layer لوحدها (Positive)
# =========================================

def test_unit_correctness_pass(perfect_adl):
    """اختبار مباشر لـ verify_correctness بالداتا الصح"""
    result = verify_correctness(perfect_adl)
    assert result["status"] == "PASSED"
    assert len(result["violations"]) == 0

def test_unit_completeness_pass(perfect_adl):
    """اختبار مباشر لـ verify_completeness بالداتا الصح"""
    result = verify_completeness(perfect_adl)
    assert result["status"] == "PASSED"
    assert len(result["issues"]) == 0

def test_unit_consistency_pass(perfect_adl, mocker):
    """اختبار مباشر لـ verify_consistency بالداتا الصح"""
    mocker.patch("application.extraction.adl.verification.consistency.validate_quality", 
                 return_value={"fan_in": {"Gateway": 0}, "fan_out": {"Gateway": 1}, "centralization_index": 0.5})
    mocker.patch("application.extraction.adl.verification.consistency.validate_domain", 
                 return_value={"communication_style": "asynchronous", "state_concentration": 1})
    
    result = verify_consistency(perfect_adl)
    assert result["status"] == "PASSED"

# =========================================
# ❌ Unit Tests: نختبر كل Layer لوحدها (Negative)
# =========================================

def test_correctness_fail():
    bad_adl = {"components": [], "relationships": []}
    result = verify_correctness(bad_adl)
    assert result["status"] == "FAILED"
    assert any(v["rule"] == "COMPONENT_EXISTENCE" for v in result["violations"])

def test_completeness_fail():
    isolated = {"components": [{"name": "A"}], "relationships": []}
    result = verify_completeness(isolated)
    assert result["status"] == "FAILED"
    assert any(i["rule"] == "NO_RELATIONSHIPS" for i in result["issues"])





def test_tc_f3_relationship_integrity():
   
    adl = {
        "components": [{"name": "AuthService"}],
        "relationships": [
            {
                "source": "AuthService",
                "target": "GhostService",  # Error: Target does not exist
                "type": "undefined-flow"   # Error: Invalid relationship type
            }
        ]
    }
    
    result = verify_correctness(adl)
    
    assert result["status"] == "FAILED"
    violation_rules = [v["rule"] for v in result["violations"]]
    
    # Asserting rules defined in the FR-17 / TC-F3 requirements
    assert "UNDEFINED_TARGET" in violation_rules
    assert "INVALID_RELATIONSHIP_TYPE" in violation_rules

# =========================================
# 🚀 Integration Tests: Runner Flow
# =========================================

def test_run_verification_full_success(perfect_adl, mocker):
    mocker.patch("application.extraction.adl.verification.consistency.validate_quality", 
                 return_value={"fan_in": {"Gateway": 0}, "fan_out": {"Gateway": 1}, "centralization_index": 0.5})
    mocker.patch("application.extraction.adl.verification.consistency.validate_domain", 
                 return_value={"communication_style": "asynchronous", "state_concentration": 1})

    result = run_verification(perfect_adl)
    assert result["status"] == "VERIFIED"

def test_run_verification_fails_at_correctness_tc_f3():
    """Ensures the runner correctly stops when TC-F3 violations are found."""
    invalid_adl = {
        "style":"Microservices",
        "components": [{"name": "Web"}, {"name": "DB"}],
        "relationships": [{"source": "Web", "target": "DB", "type": "wrong-type"}]
    }
    result = run_verification(invalid_adl)
    
    assert result["status"] == "NOT_VERIFIED"
    assert result["failed_layer"] == "correctness"
    assert result["details"]["violations"][0]["rule"] == "INVALID_RELATIONSHIP_TYPE"