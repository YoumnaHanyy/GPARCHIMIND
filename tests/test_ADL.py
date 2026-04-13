
from application.extraction.adl.adl_generator import generate_adl
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








# def test_tc_f2_components_validation():
#     req = {
#         "system_name": "E-commerce",
#         "functional_requirements": ["User login", "Place order"],
#         "non_functional_requirements": ["Fast", "Secure"],
#         "architecture_style": "microservices"
#     }
#
#     result = generate_adl(req)
#
#     components = result["components"]
#
#     # =========================
#     # 🔥 1. Check structure
#     # =========================
#     assert isinstance(components, list)
#     assert len(components) > 0
#
#     # =========================
#     # 🔥 2. Check fields exist
#     # =========================
#     for comp in components:
#         assert "name" in comp
#         assert "responsibility" in comp
#
#         assert comp["name"] != ""
#         assert comp["responsibility"] != ""
#
#     # =========================
#     # 🔥 3. Check duplicates
#     # =========================
#     names = [comp["name"] for comp in components]
#
#     assert len(names) == len(set(names))  # no duplicates
#
#
#
#
#
#
#
# def test_tc_f3_relationships_validation():
#     req = {
#         "system_name": "E-commerce",
#         "functional_requirements": ["User login", "Place order"],
#         "non_functional_requirements": ["Fast", "Secure"],
#         "architecture_style": "microservices"
#     }
#
#     result = generate_adl(req)
#
#     relationships = result["relationships"]
#     components = result["components"]
#
#     # =========================
#     # 🔥 1. basic structure
#     # =========================
#     assert isinstance(relationships, list)
#
#     # =========================
#     # 🔥 2. fields exist
#     # =========================
#     for rel in relationships:
#         assert "source" in rel
#         assert "target" in rel
#         assert "type" in rel
#
#         assert rel["source"] != ""
#         assert rel["target"] != ""
#         assert rel["type"] != ""
#
#     # =========================
#     # 🔥 3. source/target valid
#     # =========================
#     component_names = [c["name"] for c in components]
#
#     for rel in relationships:
#         assert rel["source"] in component_names
#         assert rel["target"] in component_names
#
#
#
#
# def test_tc_f4_adl_structure_validation():
#     req = {
#         "system_name": "E-commerce",
#         "functional_requirements": ["User login", "Place order"],
#         "non_functional_requirements": ["Fast", "Secure"],
#         "architecture_style": "microservices"
#     }
#
#     result = generate_adl(req)
#
#     components = result["components"]
#     relationships = result["relationships"]
#
#     component_names = [c["name"] for c in components]
#
#     # =========================
#     # 🔥 Validate relationships structure
#     # =========================
#     for rel in relationships:
#
#         # 1️⃣ source/target exist in components
#         assert rel["source"] in component_names
#         assert rel["target"] in component_names
#
#         # 2️⃣ no self-loop (optional but pro 🔥)
#         assert rel["source"] != rel["target"]
#
#         # 3️⃣ type exists and valid
#         assert "type" in rel
#         assert rel["type"] in ["data-flow", "event-flow", "sync", "async", "request"]
