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