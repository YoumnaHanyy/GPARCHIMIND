
ARCH_RULES = {

    "MICROSERVICES": {
        "Mediator": 0.3,
        "Observer": 0.25,
        "Command": 0.2,
        "Strategy": 0.15
    },

    "MONOLITH": {
        "Singleton": 0.3,
        "Factory": 0.2,
        "Builder": 0.15
    },

    "LAYERED": {
        "Factory": 0.25,
        "Builder": 0.2,
        "Strategy": 0.15
    },

    "EVENT_DRIVEN": {
        "Observer": 0.35,
        "Mediator": 0.25,
        "Command": 0.2
    },

    "CLIENT_SERVER": {
        "Mediator": 0.25,
        "Observer": 0.2,
        "Command": 0.15
    },

    "SERVICE_ORIENTED": {
        "Mediator": 0.3,
        "Command": 0.25,
        "Strategy": 0.2
    },

    "MICROKERNEL": {
        "Strategy": 0.3,
        "Command": 0.25,
        "Factory": 0.2
    },

    "COMPONENT_BASED": {
        "Factory": 0.3,
        "Strategy": 0.25,
        "Builder": 0.2
    },

    "PIPE_AND_FILTER": {
        "Command": 0.3,
        "Strategy": 0.25
    },

    "BROKER": {
        "Mediator": 0.35,
        "Observer": 0.25,
        "Command": 0.2
    },

    "PEER_TO_PEER": {
        "Mediator": 0.25,
        "Observer": 0.25
    },

    "BLACKBOARD": {
        "Mediator": 0.35,
        "Strategy": 0.25
    },

    "MVC": {
        "Observer": 0.35,
        "Strategy": 0.25,
        "Mediator": 0.2
    },

    "SPACE_BASED": {
        "Mediator": 0.3,
        "Command": 0.25
    },

    "REST": {
        "Command": 0.3,
        "Mediator": 0.25
    },

    "HEXAGONAL": {
        "Strategy": 0.3,
        "Factory": 0.25,
        "Command": 0.2
    },

    "SERVERLESS": {
        "Command": 0.35,
        "Observer": 0.25
    },

    "EVENT_BUS": {
        "Observer": 0.4,
        "Mediator": 0.3,
        "Command": 0.25
    }

}
