
from ai.ai_engine import ai_generate_architecture



def generate_adl(req):
    # -------- AI Architecture Generation --------
    ai = ai_generate_architecture(
        req["system_name"],
        req["functional_requirements"],
        req["non_functional_requirements"],
        req["architecture_style"]
    )
    
    return ai  