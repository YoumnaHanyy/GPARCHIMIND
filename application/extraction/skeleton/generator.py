from application.extraction.skeleton.styles.mvc_generator import generate_mvc
from application.extraction.skeleton.styles.layered_generator import generate_layered
from application.extraction.skeleton.styles.rest_generator import generate_rest
from application.extraction.skeleton.styles.client_server_generator import generate_client_server
from application.extraction.skeleton.styles.monolithic_generator import generate_monolithic
from application.extraction.skeleton.styles.microservices_generator import generate_microservices
from application.extraction.skeleton.styles.hexagonal_generator import generate_hexagonal
from application.extraction.skeleton.styles.soa_generator import generate_soa
from application.extraction.skeleton.styles.microkernel_generator import generate_microkernel
from application.extraction.skeleton.styles.event_bus_generator import generate_event_bus
from application.extraction.skeleton.styles.event_driven_generator import generate_event_driven
from application.extraction.skeleton.styles.serverless_generator import generate_serverless
from application.extraction.skeleton.styles.component_based_generator import generate_component_based
from application.extraction.skeleton.styles.pipe_filter_generator import generate_pipe_filter
from application.extraction.skeleton.styles.broker_generator import generate_broker
from application.extraction.skeleton.styles.space_based_generator import generate_space_based
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
    elif architecture == "MICROKERNAL (PLUG-IN)":

       return generate_microkernel(
        functional,
        nfrs,
        patterns
       )
    elif architecture == "EVENT-BUS/EVENT BROKER":

       return generate_event_bus(
         functional,
         nfrs,
         patterns
        )
    elif architecture == "EVENT-DRIVEN/ MESSAGING":

      return generate_event_driven(
        functional,
        nfrs,
        patterns
      )
    elif architecture == "SERVERLESS/FAAS":

       return generate_serverless(
        functional,
        nfrs,
        patterns
       )
    elif architecture == "COMPONENT BASED":

       return generate_component_based(
        functional,
        nfrs,
        patterns
       )
    elif architecture == "PIPE AND- FILTER":

       return generate_pipe_filter(
        functional,
        nfrs,
        patterns
       )
    elif architecture == "BROKER (MIDDLEWARE)":

       return generate_broker(
        functional,
        nfrs,
        patterns
       )
    elif architecture == "SPACE BASED":

       return generate_space_based(
        functional,
        nfrs,
        patterns
       )
    return "Architecture not supported yet"