from application.extraction.skeleton.styles.mvc_generator import generate_mvc
from application.extraction.skeleton.styles.layered_generator import generate_layered
from application.extraction.skeleton.styles.rest_generator import generate_rest

def generate_code_skeleton(
    architecture,
    functional,
    nfrs,
    patterns
):

    architecture = architecture.upper()

    print("ARCHITECTURE =", architecture)

    if architecture == "MVC (MODEL-VIEW-CONTROLLER)":
        return generate_mvc(
    functional,
    nfrs,
    patterns
)

    elif architecture == "LAYERED (N-TIER)":
        return generate_layered(
    functional,
    nfrs,
    patterns
)
    
    elif architecture == "REST/ RESOURCE-ORIENTED":
        return generate_rest(functional,nfrs,patterns)
    
    return "Architecture not supported yet"