from adl.ai_engine import ai_generate_architecture

def generate_adl(req):
    ai_result = ai_generate_architecture(
        req["functional_requirements"],
        req["non_functional_requirements"],
        req["architecture_style"]
    )

    return {
        "system": {
            "name": req["system_name"],
            "architectureStyle": req["architecture_style"]
        },
        "qualityAttributes": req["non_functional_requirements"],
        "services": ai_result["components"],
        "relationships": ai_result["relationships"]
    }
