from application.extraction.adl.adl_generator import generate_adl
from application.extraction.adl.verification.correctness import verify_correctness
from application.extraction.adl.verification.completeness import verify_completeness
import os
os.environ["HF_API_KEY"] = "fake_key"


def test_tc_f1_generate_adl():
    req = {
        "system_name": "E-commerce",
        "functional_requirements": ["User login", "Place order"],
        "non_functional_requirements": ["Fast", "Secure"],
        "architecture_style": "microservices"
    }

    result = generate_adl(req)

    # =========================
    # 🔥 STRONG ASSERTIONS
    # =========================

    # structure
    assert isinstance(result, dict)

    # keys existence
    assert "system" in result
    assert "components" in result
    assert "relationships" in result
    assert "runtime_flow" in result

    # values correctness
    assert result["system"] == "E-commerce"
    assert result["style"] == "microservices"

    # components check
    assert isinstance(result["components"], list)
    assert len(result["components"]) > 0

    first_component = result["components"][0]

    assert "name" in first_component
    assert "responsibility" in first_component

    # relationships check
    assert isinstance(result["relationships"], list)

    if result["relationships"]:
        rel = result["relationships"][0]
        assert "source" in rel
        assert "target" in rel
        assert "type" in rel

    # runtime flow
    assert isinstance(result["runtime_flow"], list)

    # source (AI or fallback)
    assert result["source"] in ["AI", "FALLBACK"]

def test_tc_f2_components_valid():
    adl = {
        "components": [
            {"name": "AuthService"},
            {"name": "OrderService"}
        ],
        "relationships": [
            {"source": "AuthService", "target": "OrderService", "type": "event-flow"}
        ]
    }

    result = verify_completeness(adl)

    assert result["status"] == "PASSED"
    assert len(result["issues"]) == 0

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