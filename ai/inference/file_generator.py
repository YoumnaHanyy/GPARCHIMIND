import json
import re

from service.srs_extractor import generate


# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------

LANG_EXT = {
    "python":     ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "java":       ".java",
    "csharp":     ".cs",
    "go":         ".go",
    "php":        ".php",
    "ruby":       ".rb",
    "kotlin":     ".kt",
    "swift":      ".swift",
}

LANG_MAIN = {
    "python":     "main.py",
    "javascript": "index.js",
    "typescript": "index.ts",
    "java":       "Main.java",
    "csharp":     "Program.cs",
    "go":         "main.go",
    "php":        "index.php",
    "ruby":       "main.rb",
    "kotlin":     "Main.kt",
    "swift":      "main.swift",
}


def get_main_file(language: str = "python") -> str:
    return LANG_MAIN.get(language, "main.py")


def _ext(language: str) -> str:
    return LANG_EXT.get(language, ".py")


def _rename(filename: str, language: str) -> str:

    # 🚫 don't touch frontend/template/static files

    ignored_extensions = [
        ".html",
        ".css",
        ".scss",
        ".sass",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".json",
        ".xml",
        ".yaml",
        ".yml"
    ]

    lower = filename.lower()

    for ext_ignore in ignored_extensions:

        if lower.endswith(ext_ignore):

            return filename

    ext = _ext(language)

    base = re.sub(
        r"\.(py|java|cs|go|php|rb|kt|swift)$",
        "",
        filename,
        flags=re.IGNORECASE,
    )

    return base + ext

def _remap_files(parsed: dict, language: str) -> dict:
    """Walk a {folder: [files] | {subfolder: [files]}} dict and rename extensions."""
    result = {}
    for folder, value in parsed.items():
        if isinstance(value, dict):
            result[folder] = _remap_files(value, language)
        else:
            result[folder] = [_rename(f, language) for f in value]
    return result


# ---------------------------------------------------------------------------
# REST / generic project files
# ---------------------------------------------------------------------------

def generate_project_files(requirements, language="python"):

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
  "routes": ["auth_routes.py"],
  "controllers": ["auth_controller.py"],
  "services": ["auth_service.py"],
  "schemas": ["auth_schema.py"]
}}
"""

    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    return _remap_files(json.loads(cleaned), language)


# ---------------------------------------------------------------------------
# NFR files  (language used only for extension remapping)
# ---------------------------------------------------------------------------

def generate_nfr_files(nfrs, language="python"):

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
Include config, middleware, utils files when needed only in the requirements.
Examples:
- security → auth middleware
- performance → cache utils
- logging → logging middleware

Return ONLY valid JSON.

Example:

{{
  "middleware": ["auth_middleware.py", "logging_middleware.py"],
  "utils": ["cache_utils.py"],
  "config": ["security_config.py", "database_config.py"]
}}
"""

    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    return _remap_files(json.loads(cleaned), language)


# ---------------------------------------------------------------------------
# Design pattern files
# ---------------------------------------------------------------------------

def generate_pattern_files(patterns, requirements, language="python"):

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
    "observer": ["event_observer.py", "subject.py"],
    "strategy": ["authentication_strategy.py", "payment_strategy.py"],
    "factory": ["service_factory.py"]
  }}
}}
"""

    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    parsed = json.loads(cleaned)

    if "patterns" in parsed and isinstance(parsed["patterns"], dict):
        remapped_inner = {
            pname: [_rename(f, language) for f in files]
            for pname, files in parsed["patterns"].items()
        }
        return {"patterns": remapped_inner}
    return parsed


# ---------------------------------------------------------------------------
# Architecture-specific file generators
# ---------------------------------------------------------------------------

def generate_mvc_project_files(requirements, language="python"):
    joined = "\n".join(requirements)
    prompt = f"""
You are a senior software architect.

Given these MVC functional requirements:

{joined}

Generate suitable MVC backend/frontend files.
Return ONLY valid JSON.
Focus only on: models, views, controllers, routes, templates.

Example:
{{
  "models": ["user_model.py"],
  "views": ["user_view.py"],
  "controllers": ["user_controller.py"],
  "routes": ["user_routes.py"],
  "templates": ["user_dashboard.html"]
}}
"""
    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    return _remap_files(json.loads(cleaned), language)


def generate_layered_project_files(requirements, language="python"):

    joined = "\n".join(requirements)

    prompt = f"""
You are a senior software architect.

Given these Layered Architecture functional requirements:

{joined}

Generate suitable files for a Layered Architecture.

Return ONLY valid JSON.

Rules:
- The presentation layer should contain:
  - controllers
  - views
  - templates
  - frontend/UI files

- Do NOT generate API-only presentation layers.

- The business layer should contain services and business logic.

- The data layer should contain repositories and persistence logic.

- The domain layer should contain entities and domain models.

- The infrastructure layer should contain external integrations and technical utilities.

Focus only on:
- presentation
- business
- data
- domain
- infrastructure

Example:

{{
  "presentation": [
    "user_controller.py",
    "login_controller.py",
    "dashboard.html",
    "login.html",
    "register.html"
  ],

  "business": [
    "user_service.py",
    "authentication_service.py"
  ],

  "data": [
    "user_repository.py",
    "database_manager.py"
  ],

  "domain": [
    "user_entity.py",
    "role_entity.py"
  ],

  "infrastructure": [
    "database_connection.py",
    "file_storage.py"
  ]
}}
"""

    output = generate(prompt)

    cleaned = re.sub(r"```json|```", "", output).strip()

    return _remap_files(json.loads(cleaned), language)

def generate_client_server_project_files(requirements, language="python"):
    joined = "\n".join(requirements)
    prompt = f"""
You are a senior software architect.

Given these Client-Server functional requirements:

{joined}

Generate suitable files for a Client-Server architecture.
Return ONLY valid JSON.
Focus only on: client, server, communication, database, shared.

Example:
{{
  "client": ["auth_client.py"],
  "server": ["auth_server.py"],
  "communication": ["socket_manager.py"],
  "database": ["database_manager.py"],
  "shared": ["shared_models.py", "constants.py"]
}}
"""
    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    return _remap_files(json.loads(cleaned), language)


def generate_monolithic_project_files(requirements, language="python"):
    joined = "\n".join(requirements)
    prompt = f"""
You are a senior software architect.

Given these Monolithic functional requirements:

{joined}

Generate suitable files for a Monolithic architecture.
Return ONLY valid JSON.
Focus only on: app (nested by feature), shared, config, database.

Example:
{{
  "app": {{
    "authentication": ["auth_controller.py", "auth_service.py", "user_model.py"],
    "video_generation": ["video_service.py", "video_model.py"]
  }},
  "shared": ["helpers.py", "constants.py"],
  "config": ["database_config.py"],
  "database": ["connection.py", "migrations.py"]
}}
"""
    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    return _remap_files(json.loads(cleaned), language)


def generate_microservices_project_files(requirements, language="python"):
    joined = "\n".join(requirements)
    prompt = f"""
You are a senior software architect.

Given these Microservices functional requirements:

{joined}

Generate suitable files for a Microservices architecture.
Return ONLY valid JSON.
Each business capability is a root-level key with a list of files.
Also include: api_gateway, communication, shared, monitoring.

Example:
{{
  "auth_service": ["auth_controller.py", "auth_service.py", "user_model.py"],
  "video_service": ["video_controller.py", "video_service.py"],
  "api_gateway": ["gateway_router.py", "api_gateway.py"],
  "communication": ["event_bus.py", "message_broker.py"],
  "shared": ["shared_dtos.py", "constants.py"],
  "monitoring": ["health_check.py", "metrics.py"]
}}
"""
    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    return _remap_files(json.loads(cleaned), language)


def generate_hexagonal_project_files(requirements, language="python"):
    joined = "\n".join(requirements)
    prompt = f"""
You are a senior software architect.

Given these Hexagonal Architecture functional requirements:

{joined}

Generate suitable files for a Hexagonal (Ports & Adapters) architecture.
Return ONLY valid JSON.
Focus only on: domain, application, ports (with input/output), adapters (with input/output), infrastructure.

Example:
{{
  "domain": ["user_entity.py", "video_entity.py"],
  "application": ["generate_video_use_case.py", "authentication_service.py"],
  "ports": {{
    "input": ["video_input_port.py"],
    "output": ["storage_output_port.py", "ai_output_port.py"]
  }},
  "adapters": {{
    "input": ["rest_controller.py"],
    "output": ["postgres_adapter.py", "openai_adapter.py"]
  }},
  "infrastructure": ["database_config.py", "event_bus.py"]
}}
"""
    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    return _remap_files(json.loads(cleaned), language)


def generate_soa_project_files(requirements, language="python"):
    joined = "\n".join(requirements)
    prompt = f"""
You are a senior software architect.

Given these SOA functional requirements:

{joined}

Generate suitable files for a Service-Oriented Architecture.
Return ONLY valid JSON.
Focus only on: service_providers, service_consumers, service_registry, service_contracts, orchestration, messaging, adapters, shared.

Example:
{{
  "service_providers": ["authentication_provider.py"],
  "service_consumers": ["video_consumer.py"],
  "service_registry": ["service_registry.py"],
  "service_contracts": ["authentication_contract.py"],
  "orchestration": ["workflow_orchestrator.py"],
  "messaging": ["message_broker.py", "event_queue.py"],
  "adapters": ["openai_adapter.py"],
  "shared": ["shared_dtos.py", "constants.py"]
}}
"""
    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    return _remap_files(json.loads(cleaned), language)


def generate_microkernel_project_files(requirements, language="python"):
    joined = "\n".join(requirements)
    prompt = f"""
You are a senior software architect.

Given these Microkernel functional requirements:

{joined}

Generate suitable files for a Microkernel architecture.
Return ONLY valid JSON.
Focus only on: core, plugins (nested dict), interfaces, services, shared, config.

Example:
{{
  "core": ["kernel.py", "plugin_manager.py"],
  "plugins": {{
    "authentication_plugin": ["auth_plugin.py"],
    "video_plugin": ["video_plugin.py"]
  }},
  "interfaces": ["plugin_interface.py"],
  "services": ["kernel_service.py"],
  "shared": ["helpers.py", "constants.py"],
  "config": ["system_config.py", "plugin_config.py"]
}}
"""
    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    return _remap_files(json.loads(cleaned), language)


def generate_event_bus_project_files(requirements, language="python"):
    joined = "\n".join(requirements)
    prompt = f"""
You are a senior software architect.

Given these Event-Bus functional requirements:

{joined}

Generate suitable files for an Event-Bus architecture.
Return ONLY valid JSON.
Focus only on: producers, consumers, events, broker, channels, handlers, shared, monitoring.

Example:
{{
  "producers": ["video_event_producer.py"],
  "consumers": ["analytics_consumer.py"],
  "events": ["video_generated_event.py"],
  "broker": ["event_bus.py", "message_broker.py"],
  "channels": ["video_channel.py"],
  "handlers": ["video_event_handler.py"],
  "shared": ["shared_dtos.py", "constants.py"],
  "monitoring": ["event_monitor.py", "tracing.py"]
}}
"""
    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    return _remap_files(json.loads(cleaned), language)


def generate_event_driven_project_files(requirements, language="python"):
    joined = "\n".join(requirements)
    prompt = f"""
You are a senior software architect.

Given these Event-Driven functional requirements:

{joined}

Generate suitable files for an Event-Driven architecture.
Return ONLY valid JSON.
Focus only on: event_producers, event_consumers, event_channels, event_processing, messaging_infrastructure, shared.

Example:
{{
  "event_producers": ["video_event_producer.py"],
  "event_consumers": ["analytics_event_consumer.py"],
  "event_channels": ["video_channel.py"],
  "event_processing": ["video_event_processor.py"],
  "messaging_infrastructure": ["event_bus.py", "message_router.py"],
  "shared": ["shared_dtos.py", "constants.py"]
}}
"""
    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    return _remap_files(json.loads(cleaned), language)


def generate_serverless_project_files(requirements, language="python"):
    joined = "\n".join(requirements)
    prompt = f"""
You are a senior software architect.

Given these Serverless functional requirements:

{joined}

Generate suitable files for a Serverless/FaaS architecture.
Return ONLY valid JSON.
Focus only on: functions, triggers, events, integrations, workflows, shared, deployment.

Example:
{{
  "functions": ["auth_function.py", "video_generation_function.py"],
  "triggers": ["api_trigger.py", "queue_trigger.py"],
  "events": ["video_generated_event.py"],
  "integrations": ["openai_integration.py"],
  "workflows": ["video_processing_workflow.py"],
  "shared": ["shared_dtos.py", "constants.py"],
  "deployment": ["deployment_pipeline.py"]
}}
"""
    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    return _remap_files(json.loads(cleaned), language)


def generate_component_based_project_files(requirements, language="python"):
    joined = "\n".join(requirements)
    prompt = f"""
You are a senior software architect.

Given these Component-Based functional requirements:

{joined}

Generate suitable files for a Component-Based architecture.
Return ONLY valid JSON.
Focus only on: components, interfaces, connectors, services, shared, configuration.

Example:
{{
  "components": ["authentication_component.py", "video_processing_component.py"],
  "interfaces": ["authentication_interface.py"],
  "connectors": ["component_connector.py", "event_connector.py"],
  "services": ["component_registry_service.py"],
  "shared": ["shared_dtos.py", "constants.py"],
  "configuration": ["component_config.py", "registry_config.py"]
}}
"""
    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    return _remap_files(json.loads(cleaned), language)


def generate_pipe_filter_project_files(requirements, language="python"):
    joined = "\n".join(requirements)
    prompt = f"""
You are a senior software architect.

Given these Pipe-and-Filter functional requirements:

{joined}

Generate suitable files for a Pipe-and-Filter architecture.
Return ONLY valid JSON.
Focus only on: filters, pipes, pipelines, processors, shared, monitoring.

Example:
{{
  "filters": ["validation_filter.py", "text_cleaning_filter.py"],
  "pipes": ["data_pipe.py", "stream_pipe.py"],
  "pipelines": ["analytics_pipeline.py"],
  "processors": ["pipeline_executor.py"],
  "shared": ["shared_dtos.py", "constants.py"],
  "monitoring": ["pipeline_logger.py", "pipeline_monitor.py"]
}}
"""
    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    return _remap_files(json.loads(cleaned), language)


def generate_broker_project_files(requirements, language="python"):
    joined = "\n".join(requirements)
    prompt = f"""
You are a senior software architect.

Given these Broker functional requirements:

{joined}

Generate suitable files for a Broker architecture.
Return ONLY valid JSON.
Focus only on: clients, brokers, servers, communication, messaging, shared.

Example:
{{
  "clients": ["user_client.py"],
  "brokers": ["service_broker.py", "request_router.py"],
  "servers": ["authentication_server.py", "video_server.py"],
  "communication": ["protocol_manager.py"],
  "messaging": ["message_dispatcher.py"],
  "shared": ["shared_dtos.py", "contracts.py"]
}}
"""
    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    return _remap_files(json.loads(cleaned), language)


def generate_space_based_project_files(requirements, language="python"):
    joined = "\n".join(requirements)
    prompt = f"""
You are a senior software architect.

Given these Space-Based functional requirements:

{joined}

Generate suitable files for a Space-Based architecture.
Return ONLY valid JSON.
Focus only on: processing_units, data_grid, messaging_grid, virtualized_middleware, deployment_manager, shared.

Example:
{{
  "processing_units": ["video_processing_unit.py"],
  "data_grid": ["distributed_cache.py", "replication_manager.py"],
  "messaging_grid": ["event_dispatcher.py"],
  "virtualized_middleware": ["load_balancer.py"],
  "deployment_manager": ["scaling_manager.py"],
  "shared": ["shared_dtos.py", "constants.py"]
}}
"""
    output = generate(prompt)
    cleaned = re.sub(r"```json|```", "", output).strip()
    return _remap_files(json.loads(cleaned), language)