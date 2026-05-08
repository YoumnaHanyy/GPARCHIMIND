from application.extraction.skeleton.styles.mvc_generator import generate_mvc
from application.extraction.skeleton.styles.layered_generator import generate_layered
from application.extraction.skeleton.styles.rest_generator import generate_rest
from application.extraction.skeleton.styles.client_server_generator import generate_client_server
from application.extraction.skeleton.styles.monolithic_generator import generate_monolithic
from application.extraction.skeleton.styles.microservices_generator import generate_microservices
from application.extraction.skeleton.styles.hexagonal_generator import generate_hexagonal
from application.extraction.skeleton.styles.soa_generator import generate_soa

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
    elif architecture == "CLIENT-SERVER":

        return generate_client_server(
          functional,
          nfrs,
          patterns
        )
    elif architecture == "MONOLITHIC":

        return generate_monolithic(
          functional,
          nfrs,
          patterns
        )
    elif architecture == "MICROSERVICES":

        return generate_microservices(
           functional,
           nfrs,
           patterns
        )
    elif architecture == "REST/ RESOURCE-ORIENTED":
        return generate_rest(functional,nfrs,patterns)
    elif architecture == "HEXAGONAL (PORTS& ADAPTERS)":

       return generate_hexagonal(
        functional,
        nfrs,
        patterns
       )
    elif architecture == "SERVICE ORIENTED (SOA)":

       return generate_soa(
        functional,
        nfrs,
        patterns
        )
    return "Architecture not supported yet"