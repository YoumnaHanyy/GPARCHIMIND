import json
import re

from service.srs_extractor import generate


def generate_project_files(requirements):

    joined_requirements = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Given these REST API functional requirements:

{joined_requirements}

Generate suitable backend files.
Generate ONLY business logic files.

Do NOT generate:
- middleware
- config
- monitoring
- utils

Focus only on if needed in functional requirements:
- routes
- controllers
- services
- schemas
- models
- repositories

Return ONLY valid JSON.

Example:

{{
  "routes": [
    "auth_routes.py"
  ],

  "controllers": [
    "auth_controller.py"
  ],

  "services": [
    "auth_service.py"
  ],

  "schemas": [
    "auth_schema.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

def generate_nfr_files(nfrs):

    joined_nfrs = "\n".join([
        nfr.get("type", "")
        for nfr in nfrs
        if isinstance(nfr, dict)
    ])

    prompt = f"""
You are a senior software architect.

Given these non-functional requirements:

{joined_nfrs}

Generate supporting backend files.
Include config, middleware,utils files when needed only in the requirements.
Examples:
- security → auth middleware
- performance → cache utils
- logging → logging middleware

Return ONLY valid JSON.

Example:

{{
  "middleware": [
    "auth_middleware.py",
    "logging_middleware.py"
  ],

  "utils": [
    "cache_utils.py"
  ],

    "config": [
    "security_config.py",
    "database_config.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

def generate_pattern_files(patterns,requirements):
    
    joined_patterns = "\n".join([
      pattern.get("pattern", "")
      for pattern in patterns
      if isinstance(pattern, dict)
    ])
    joined_requirements = "\n".join(requirements)
    prompt = f"""
You are a senior software architect.

Given these recommended design patterns:

{joined_patterns}
And these project requirements:

{joined_requirements}

Generate ONLY design pattern structure.

Return ONLY valid JSON.

Rules:
- Put everything inside a main folder called "patterns".
- Inside "patterns", create subfolders for each design pattern.
- Each pattern folder should contain suitable implementation files.
- Do NOT generate routes, services, repositories, controllers, or models.
- Focus ONLY on pattern implementation files.

Example:

{{
  "patterns": {{
    
    "observer": [
      "event_observer.py",
      "subject.py"
    ],

    "strategy": [
      "authentication_strategy.py",
      "payment_strategy.py"
    ],

    "factory": [
      "service_factory.py"
    ]
  }}
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

def generate_mvc_project_files(requirements):

    joined_requirements = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Given these MVC functional requirements:

{joined_requirements}

Generate suitable MVC backend/frontend files.

Return ONLY valid JSON.

Focus only on:
- models
- views
- controllers
- routes
- templates

Example:

{{
  "models": [
    "user_model.py"
  ],

  "views": [
    "user_view.py"
  ],

  "controllers": [
    "user_controller.py"
  ],

  "routes": [
    "user_routes.py"
  ],

  "templates": [
    "user_dashboard.html"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

def generate_layered_project_files(requirements):

    joined_requirements = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Given these Layered Architecture functional requirements:

{joined_requirements}

Generate suitable files for a layered architecture.

Return ONLY valid JSON.

Focus only on:
- presentation
- business
- data
- domain
- infrastructure

Example:

{{
  "presentation": [
    "auth_api.py",
    "video_api.py"
  ],

  "business": [
    "auth_service.py",
    "video_service.py"
  ],

  "data": [
    "user_repository.py",
    "video_repository.py"
  ],

  "domain": [
    "user_entity.py",
    "video_entity.py"
  ],

  "infrastructure": [
    "database_connection.py",
    "storage_manager.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

def generate_client_server_project_files(requirements):

    joined_requirements = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Based on Client-Server architecture principles and logical layering.

The architecture should separate:
- client responsibilities
- server responsibilities
- communication handling
- shared resources
- database management

Given these Client-Server functional requirements:

{joined_requirements}

Generate suitable files for a Client-Server architecture.

Return ONLY valid JSON.

Focus only on:
- client
- server
- communication
- database
- shared

Rules:
- client → UI, frontend interaction, client-side logic
- server → business logic and request processing
- communication → sockets, APIs, request handlers
- database → database access and storage
- shared → shared DTOs, models, constants

Example:

{{
  "client": [
    "auth_client.py",
    "video_client.py"
  ],

  "server": [
    "auth_server.py",
    "video_server.py"
  ],

  "communication": [
    "socket_manager.py",
    "request_handler.py"
  ],

  "database": [
    "database_manager.py",
    "storage_handler.py"
  ],

  "shared": [
    "shared_models.py",
    "constants.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

def generate_monolithic_project_files(requirements):

    joined_requirements = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Based on monolithic architecture principles.

A monolithic system should:
- be a single deployable application
- organize the system around business modules/features
- keep all components tightly integrated
- centralize shared logic and configuration

Given these Monolithic functional requirements:

{joined_requirements}

Generate suitable files and folders for a Monolithic architecture.

Return ONLY valid JSON.

Focus only on:
- app
- shared
- config
- database

Rules:
- app contains feature modules
- each feature module may contain:
  - controllers
  - services
  - models
  - repositories
- shared contains common utilities/helpers
- config contains centralized configuration
- database contains database setup/access

Examples of shared files:
- helpers.py
- validators.py
- logger.py
- constants.py

Examples of config files:
- database_config.py
- security_config.py
- logging_config.py

Examples of database files:
- connection.py
- migrations.py
- db_session.py

Example:

{{
  "app": {{

    "authentication": [
      "auth_controller.py",
      "auth_service.py",
      "user_model.py",
      "user_repository.py"
    ],

    "video_generation": [
      "video_service.py",
      "video_generator.py",
      "video_model.py"
    ],

    "analytics": [
      "analytics_service.py",
      "analytics_model.py"
    ]
  }},

  "shared": [
    "helpers.py",
    "validators.py",
    "logger.py",
    "constants.py"
  ],

  "config": [
    "database_config.py",
    "security_config.py",
    "logging_config.py"
  ],

  "database": [
    "connection.py",
    "migrations.py",
    "db_session.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

def generate_microservices_project_files(requirements):

    joined_requirements = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Based on Microservices architecture principles.

A microservices system should:
- split the system into independent services
- isolate each business capability into its own microservice
- allow independent deployment and scaling
- separate databases/configuration per service
- support inter-service communication
- organize services as independent root-level components

Given these Microservices functional requirements:

{joined_requirements}

Generate suitable files and folders for a Microservices architecture.

Return ONLY valid JSON.

Focus only on:
- independent microservices
- api_gateway
- communication
- shared
- monitoring

Rules:
- Each business capability should be generated as an independent root-level microservice.
- Each microservice may contain:
  - controllers
  - services
  - models
  - repositories
  - configs
  - database files
- api_gateway handles routing/API gateway logic
- communication handles messaging/events/service communication
- shared contains shared DTOs/constants only
- monitoring contains tracing/logging/health monitoring

Examples of communication files:
- event_bus.py
- message_broker.py
- service_registry.py

Examples of monitoring files:
- tracing.py
- health_check.py
- metrics.py

Example:

{{
  "auth_service": [
    "auth_controller.py",
    "auth_service.py",
    "user_model.py",
    "user_repository.py",
    "database_config.py"
  ],

  "video_service": [
    "video_controller.py",
    "video_service.py",
    "video_model.py",
    "video_repository.py",
    "video_storage.py"
  ],

  "analytics_service": [
    "analytics_controller.py",
    "analytics_service.py",
    "analytics_model.py"
  ],

  "notification_service": [
    "notification_controller.py",
    "notification_service.py"
  ],

  "api_gateway": [
    "gateway_router.py",
    "api_gateway.py"
  ],

  "communication": [
    "event_bus.py",
    "message_broker.py",
    "service_registry.py"
  ],

  "shared": [
    "shared_dtos.py",
    "constants.py"
  ],

  "monitoring": [
    "health_check.py",
    "metrics.py",
    "tracing.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

def generate_hexagonal_project_files(requirements):

    joined_requirements = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Based on Hexagonal Architecture (Ports and Adapters) principles.

The architecture should:
- isolate business/domain logic
- separate application core from infrastructure
- use ports for abstraction
- use adapters for external communication
- keep the domain independent

Given these Hexagonal Architecture functional requirements:

{joined_requirements}

Generate suitable files and folders for a Hexagonal architecture.

Return ONLY valid JSON.

Focus only on:
- domain
- application
- ports
- adapters
- infrastructure

Rules:
- domain contains entities and core business rules
- application contains use cases and application services
- ports define interfaces/contracts
- adapters implement ports
- infrastructure contains external integrations/configuration

Example:

{{
  "domain": [
    "user_entity.py",
    "video_entity.py"
  ],

  "application": [
    "generate_video_use_case.py",
    "authentication_service.py"
  ],

  "ports": {{
    
    "input": [
      "video_input_port.py"
    ],

    "output": [
      "storage_output_port.py",
      "ai_output_port.py"
    ]
  }},

  "adapters": {{

    "input": [
      "rest_controller.py",
      "graphql_controller.py"
    ],

    "output": [
      "postgres_adapter.py",
      "openai_adapter.py"
    ]
  }},

  "infrastructure": [
    "database_config.py",
    "event_bus.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

def generate_soa_project_files(requirements):

    joined_requirements = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Based on Service-Oriented Architecture (SOA) principles from research literature.

The architecture should follow the core SOA components:
- Service Providers
- Service Consumers
- Service Registry
- Service Contracts
- Messaging Infrastructure
- Service Orchestration

Given these SOA functional requirements:

{joined_requirements}

Generate suitable files and folders for a Service-Oriented Architecture.

Return ONLY valid JSON.

Focus only on:
- service_providers
- service_consumers
- service_registry
- service_contracts
- orchestration
- messaging
- adapters
- shared

Rules:
- service_providers contain reusable enterprise services
- service_consumers contain clients/consumers of services
- service_registry contains service discovery/registry logic
- service_contracts contain service interfaces/contracts
- orchestration contains workflow coordination/services composition
- messaging contains brokers, queues, and event communication
- adapters contain external system integrations
- shared contains shared DTOs/constants

Example:

{{
  "service_providers": [
    "authentication_provider.py",
    "video_processing_provider.py",
    "analytics_provider.py"
  ],

  "service_consumers": [
    "video_consumer.py",
    "analytics_consumer.py"
  ],

  "service_registry": [
    "service_registry.py",
    "service_discovery.py"
  ],

  "service_contracts": [
    "authentication_contract.py",
    "video_contract.py"
  ],

  "orchestration": [
    "workflow_orchestrator.py",
    "service_composer.py"
  ],

  "messaging": [
    "message_broker.py",
    "event_queue.py"
  ],

  "adapters": [
    "openai_adapter.py",
    "payment_adapter.py"
  ],

  "shared": [
    "shared_dtos.py",
    "constants.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

def generate_microkernel_project_files(requirements):

    joined_requirements = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Based on Microkernel Architecture principles.

The architecture should:
- contain a minimal core system
- support extensibility through plugins
- isolate optional features as plugins
- use interfaces/contracts between core and plugins

Given these Microkernel functional requirements:

{joined_requirements}

Generate suitable files and folders for a Microkernel architecture.

Return ONLY valid JSON.

Focus only on:
- core
- plugins
- interfaces
- services
- shared
- config

Rules:
- core contains the minimal system kernel
- plugins contain optional/extensible features
- interfaces contain plugin contracts/interfaces
- services contain internal core services
- shared contains common utilities/constants
- config contains centralized configuration

Example:

{{
  "core": [
    "kernel.py",
    "plugin_manager.py",
    "system_loader.py"
  ],

  "plugins": {{

    "authentication_plugin": [
      "auth_plugin.py"
    ],

    "video_plugin": [
      "video_plugin.py"
    ],

    "analytics_plugin": [
      "analytics_plugin.py"
    ]
  }},

  "interfaces": [
    "plugin_interface.py",
    "service_interface.py"
  ],

  "services": [
    "kernel_service.py",
    "plugin_registry_service.py"
  ],

  "shared": [
    "helpers.py",
    "constants.py"
  ],

  "config": [
    "system_config.py",
    "plugin_config.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

def generate_event_bus_project_files(requirements):

    joined_requirements = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Based on Event-Bus / Event-Broker architecture principles.

The architecture should:
- support asynchronous communication
- use producers and consumers
- exchange events through a broker/event bus
- decouple services/components
- organize communication around events/topics

Given these Event-Bus functional requirements:

{joined_requirements}

Generate suitable files and folders for an Event-Bus architecture.

Return ONLY valid JSON.

Focus only on:
- producers
- consumers
- events
- broker
- channels
- handlers
- shared
- monitoring

Rules:
- producers publish events
- consumers subscribe to events
- events define event payloads/types
- broker handles event routing/distribution
- channels define topics/event streams
- handlers process events
- shared contains common DTOs/constants
- monitoring contains tracing/logging/event monitoring

Example:

{{
  "producers": [
    "video_event_producer.py",
    "notification_producer.py"
  ],

  "consumers": [
    "analytics_consumer.py",
    "email_consumer.py"
  ],

  "events": [
    "video_generated_event.py",
    "user_registered_event.py"
  ],

  "broker": [
    "event_bus.py",
    "message_broker.py"
  ],

  "channels": [
    "video_channel.py",
    "notification_channel.py"
  ],

  "handlers": [
    "video_event_handler.py",
    "notification_handler.py"
  ],

  "shared": [
    "shared_dtos.py",
    "constants.py"
  ],

  "monitoring": [
    "event_monitor.py",
    "tracing.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

def generate_event_driven_project_files(requirements):

    joined_requirements = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Based on Event-Driven / Messaging architecture principles.

The architecture should:
- support asynchronous communication
- organize communication around events
- separate event producers and consumers
- support distributed messaging systems
- process events independently

Given these Event-Driven functional requirements:

{joined_requirements}

Generate suitable files and folders for an Event-Driven architecture.

Return ONLY valid JSON.

Focus only on:
- event_producers
- event_consumers
- event_channels
- event_processing
- messaging_infrastructure
- shared

Rules:
- event_producers publish events/messages
- event_consumers consume events/messages
- event_channels represent queues/topics/streams
- event_processing contains handlers/processors/workflows
- messaging_infrastructure contains brokers/dispatchers/routers
- shared contains shared DTOs/constants

Example:

{{
  "event_producers": [
    "video_event_producer.py",
    "notification_event_producer.py"
  ],

  "event_consumers": [
    "analytics_event_consumer.py",
    "email_event_consumer.py"
  ],

  "event_channels": [
    "video_channel.py",
    "notification_topic.py"
  ],

  "event_processing": [
    "video_event_processor.py",
    "notification_workflow.py"
  ],

  "messaging_infrastructure": [
    "event_bus.py",
    "message_router.py",
    "broker_manager.py"
  ],

  "shared": [
    "shared_dtos.py",
    "constants.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

def generate_serverless_project_files(requirements):

    joined_requirements = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Based on Traditional Serverless / Function-as-a-Service (FaaS) architecture principles.

The architecture should:
- organize the system around stateless functions
- support event-driven execution
- use cloud/serverless integrations
- support triggers and workflows
- minimize infrastructure management

Given these Serverless functional requirements:

{joined_requirements}

Generate suitable files and folders for a traditional serverless architecture.

Return ONLY valid JSON.

Focus only on:
- functions
- triggers
- events
- integrations
- workflows
- shared
- deployment

Rules:
- functions contain stateless serverless functions
- triggers contain API/event/queue triggers
- events contain event payload definitions
- integrations contain external/cloud integrations
- workflows contain orchestration/pipelines
- shared contains shared DTOs/constants
- deployment contains deployment/configuration files

Example:

{{
  "functions": [
    "auth_function.py",
    "video_generation_function.py",
    "notification_function.py"
  ],

  "triggers": [
    "api_trigger.py",
    "queue_trigger.py"
  ],

  "events": [
    "video_generated_event.py",
    "user_registered_event.py"
  ],

  "integrations": [
    "openai_integration.py",
    "stripe_integration.py"
  ],

  "workflows": [
    "video_processing_workflow.py"
  ],

  "shared": [
    "shared_dtos.py",
    "constants.py"
  ],

  "deployment": [
    "deployment_pipeline.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

def generate_component_based_project_files(requirements):

    joined_requirements = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Based on Component-Based Software Engineering (CBSE) principles.

The architecture should:
- organize the system around reusable components
- separate components using interfaces/contracts
- support modularity and composability
- support component communication/connectors

Given these Component-Based functional requirements:

{joined_requirements}

Generate suitable files and folders for a Component-Based architecture.

Return ONLY valid JSON.

Focus only on:
- components
- interfaces
- connectors
- services
- shared
- configuration

Rules:
- components contain reusable business components
- interfaces contain component contracts/interfaces
- connectors manage communication between components
- services contain supporting/internal services
- shared contains shared DTOs/constants
- configuration contains component registration/configuration

Example:

{{
  "components": [
    "authentication_component.py",
    "video_processing_component.py",
    "analytics_component.py"
  ],

  "interfaces": [
    "authentication_interface.py",
    "video_interface.py"
  ],

  "connectors": [
    "component_connector.py",
    "event_connector.py"
  ],

  "services": [
    "component_registry_service.py"
  ],

  "shared": [
    "shared_dtos.py",
    "constants.py"
  ],

  "configuration": [
    "component_config.py",
    "registry_config.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

def generate_pipe_filter_project_files(requirements):

    joined_requirements = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Based on Pipe-and-Filter architecture principles.

The architecture should:
- organize the system around independent filters
- process data through sequential stages
- use pipes for communication between filters
- support reusable processing pipelines

Given these Pipe-and-Filter functional requirements:

{joined_requirements}

Generate suitable files and folders for a Pipe-and-Filter architecture.

Return ONLY valid JSON.

Focus only on:
- filters
- pipes
- pipelines
- processors
- shared
- monitoring

Rules:
- filters contain independent transformation stages
- pipes contain communication/dataflow channels
- pipelines contain ordered workflow pipelines
- processors contain execution/processing logic
- shared contains shared DTOs/constants
- monitoring contains tracing/logging/error tracking

Example:

{{
  "filters": [
    "validation_filter.py",
    "text_cleaning_filter.py",
    "image_processing_filter.py"
  ],

  "pipes": [
    "data_pipe.py",
    "stream_pipe.py"
  ],

  "pipelines": [
    "analytics_pipeline.py",
    "image_pipeline.py"
  ],

  "processors": [
    "pipeline_executor.py",
    "data_processor.py"
  ],

  "shared": [
    "shared_dtos.py",
    "constants.py"
  ],

  "monitoring": [
    "pipeline_logger.py",
    "pipeline_monitor.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

def generate_broker_project_files(requirements):

    joined_requirements = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Based on Broker Architecture principles.

The architecture should:
- separate clients and servers
- use brokers for request mediation
- support distributed communication
- support message-based interaction

Given these Broker functional requirements:

{joined_requirements}

Generate suitable files and folders for a Broker architecture.

Return ONLY valid JSON.

Focus only on:
- clients
- brokers
- servers
- communication
- messaging
- shared

Rules:
- clients request distributed services
- brokers mediate and route requests
- servers provide services/resources
- communication contains communication/protocol logic
- messaging contains message transport/event messaging
- shared contains shared DTOs/contracts/constants

Example:

{{
  "clients": [
    "user_client.py",
    "analytics_client.py"
  ],

  "brokers": [
    "service_broker.py",
    "request_router.py"
  ],

  "servers": [
    "authentication_server.py",
    "video_server.py"
  ],

  "communication": [
    "protocol_manager.py",
    "network_connector.py"
  ],

  "messaging": [
    "message_dispatcher.py",
    "event_transport.py"
  ],

  "shared": [
    "shared_dtos.py",
    "contracts.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)

# ai/inference/file_generator.py

def generate_space_based_project_files(requirements):

    joined_requirements = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Based on Space-Based Architecture principles.

The architecture should:
- support high scalability
- distribute processing across processing units
- use distributed in-memory data grids
- support asynchronous messaging
- avoid centralized database bottlenecks

Given these Space-Based functional requirements:

{joined_requirements}

Generate suitable files and folders for a Space-Based architecture.

Return ONLY valid JSON.

Focus only on:
- processing_units
- data_grid
- messaging_grid
- virtualized_middleware
- deployment_manager
- shared

Rules:
- processing_units contain distributed business logic units
- data_grid contains distributed caching/data replication logic
- messaging_grid contains async communication/event messaging
- virtualized_middleware contains runtime/load balancing middleware
- deployment_manager contains scaling/deployment management
- shared contains shared DTOs/constants/contracts

Example:

{{
  "processing_units": [
    "video_processing_unit.py",
    "analytics_processing_unit.py"
  ],

  "data_grid": [
    "distributed_cache.py",
    "replication_manager.py"
  ],

  "messaging_grid": [
    "event_dispatcher.py",
    "message_router.py"
  ],

  "virtualized_middleware": [
    "load_balancer.py",
    "runtime_manager.py"
  ],

  "deployment_manager": [
    "scaling_manager.py",
    "deployment_controller.py"
  ],

  "shared": [
    "shared_dtos.py",
    "constants.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return json.loads(cleaned)