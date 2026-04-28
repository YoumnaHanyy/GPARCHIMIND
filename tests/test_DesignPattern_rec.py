from ai.inference.feature_extractor import extract_features
from ai.inference.archi_rulesMain import run_design_patterns
from ai.inference.archi_rulesMain import run_inference

def test_tc_g1_extract_features(mocker):

    requirements = [
        {"description": "System should support plugins and extensibility"},
        {"description": "System processes data in workflow steps"}
    ]

    # 🔥 mock DB call
    mocker.patch(
        "ai.inference.feature_extractor.load_selected_architecture",
        return_value="MICROSERVICES"
    )

    # 🔥 mock extraction input if needed
    mocker.patch(
        "ai.inference.feature_extractor.load_requirements",
        return_value=requirements
    )

    features = extract_features("test_project")  # ✅ مش requirements

    assert isinstance(features, dict)
    assert len(features) > 0

def test_tc_g2_match_features_with_patterns():

    input_data = {
        "architecture": "MICROSERVICES",
        "features": {
            "EVENT_DRIVEN": {
                "supported": True,
                "confidence": 0.8,
                "evidence": ["event driven system"]
            },
            "HIGH_SCALABILITY": {
                "supported": True,
                "confidence": 0.9,
                "evidence": ["high load system"]
            },
            "LOW_LATENCY": {
                "supported": True,
                "confidence": 0.6,
                "evidence": ["fast response"]
            }
        }
    }

    result = run_inference(input_data)

    assert "top_patterns" in result
    assert isinstance(result["top_patterns"], list)
    assert len(result["top_patterns"]) > 0

def test_tc_g3_display_top_patterns():

    input_data = {
        "architecture": "LAYERED",
        "features": {
            "MODULARITY": {
                "supported": True,
                "confidence": 0.9,
                "evidence": ["modular architecture"]
            },
            "HIGH_MAINTAINABILITY": {
                "supported": True,
                "confidence": 0.8,
                "evidence": ["easy to maintain"]
            }
        }
    }

    result = run_design_patterns(input_data)

    assert "top_patterns" in result
    assert len(result["top_patterns"]) > 0
    assert len(result["top_patterns"]) <= 3