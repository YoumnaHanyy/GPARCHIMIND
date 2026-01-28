from huggingface_hub import InferenceClient
import json



# ================= LLM CONFIG =================





# ================= LLM HELPERS =================

def ask_llm(prompt: str, temperature=0.2):
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a senior production software architect."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=900,
        temperature=temperature
    )
    return response["choices"][0]["message"]["content"]


def extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start:end + 1])
    raise ValueError("Invalid JSON")

# ================= SAFE JSON GENERATION =================

def robust_llm_json(prompt, retries=4):
    """
    Tries hard to get valid JSON from the LLM.
    If it fails, we raise an error (no silent fallback).
    """
    last_error = None

    for _ in range(retries):
        try:
            return extract_json(ask_llm(prompt))
        except Exception as e:
            last_error = e
            prompt = f"""
RETURN ONLY VALID JSON.
NO explanations.
NO markdown.
NO comments.

TASK:
{prompt}
"""

    raise RuntimeError(f"LLM failed after {retries} attempts: {last_error}")

# ================= FALLBACK (STYLE-AWARE) =================

def fallback_components(style):
    style = style.lower()

    if "event" in style:
        return [
            {"name": "Order Service", "responsibility": "Manage orders"},
            {"name": "Payment Service", "responsibility": "Process payments"},
            {"name": "Notification Service", "responsibility": "Send notifications"},
            {"name": "Message Broker", "responsibility": "Asynchronous communication"},
            {"name": "Order Database", "responsibility": "Persist order data"}
        ]

    if "micro" in style:
        return [
            {"name": "API Gateway", "responsibility": "Route client requests"},
            {"name": "User Service", "responsibility": "Manage users"},
            {"name": "Order Service", "responsibility": "Manage orders"},
            {"name": "Order Database", "responsibility": "Persist order data"}
        ]

    # default layered
    return [
        {"name": "API", "responsibility": "Handle client requests"},
        {"name": "Service", "responsibility": "Business logic"},
        {"name": "Database", "responsibility": "Persistent storage"}
    ]


def fallback_relationships(style):
    style = style.lower()

    if "event" in style:
        return [
            {"source": "Order Service", "target": "Message Broker", "type": "event-flow"},
            {"source": "Payment Service", "target": "Message Broker", "type": "event-flow"},
            {"source": "Message Broker", "target": "Notification Service", "type": "event-flow"},
            {"source": "Order Service", "target": "Order Database", "type": "data-flow"}
        ]

    if "micro" in style:
        return [
            {"source": "API Gateway", "target": "User Service", "type": "data-flow"},
            {"source": "API Gateway", "target": "Order Service", "type": "data-flow"},
            {"source": "Order Service", "target": "Order Database", "type": "data-flow"}
        ]

    # default layered
    return [
        {"source": "API", "target": "Service", "type": "data-flow"},
        {"source": "Service", "target": "Database", "type": "data-flow"}
    ]

# ================= STYLE-AWARE RULES =================

def style_production_intents(style: str):
    style = style.lower()

    if "event" in style:
        return {
            "deployment": {"scaling": "horizontal", "availability": "multi-region"},
            "security": {"authentication": "service-to-service", "authorization": "RBAC"},
            "resilience": {"patterns": ["Retry", "CircuitBreaker"], "delivery": "at-least-once", "ordering": False}
        }

    if "layered" in style:
        return {
            "deployment": {"scaling": "vertical", "availability": "single-region"},
            "security": {"authentication": "centralized", "authorization": "RBAC"},
            "resilience": {"patterns": ["Retry"], "delivery": "at-most-once", "ordering": True}
        }

    return {
        "deployment": {"scaling": "horizontal", "availability": "multi-region"},
        "security": {"authentication": "service-to-service", "authorization": "RBAC"},
        "resilience": {"patterns": ["Retry", "CircuitBreaker"], "delivery": "at-least-once", "ordering": False}
    }

# ================= ARCHITECTURE STEPS =================

def extract_decisions(system, frs, nfrs, style):
    prompt = f"""
System: {system}
Architecture Style: {style}

Functional Requirements:
{frs}

Non-Functional Requirements:
{nfrs}

Return JSON:
{{ "decisions": [{{ "name": "...", "rationale": "..." }}] }}
"""
    return robust_llm_json(prompt).get("decisions", [])


def generate_components(system, frs):
    prompt = f"""
System: {system}

Functional Requirements:
{frs}

Return JSON:
{{ "components": [{{ "name": "...", "responsibility": "..." }}] }}
"""
    return robust_llm_json(prompt).get("components", [])


def generate_relationships(components):
    prompt = f"""
Components:
{json.dumps(components, indent=2)}

Return JSON:
{{ "relationships": [{{ "source": "...", "target": "...", "type": "data-flow | event-flow" }}] }}
"""
    return robust_llm_json(prompt).get("relationships", [])


def critique(components, relationships, nfrs):
    prompt = f"""
Components:
{components}

Relationships:
{relationships}

NFRs:
{nfrs}

Return JSON:
{{ "issues": [] }}
"""
    return robust_llm_json(prompt).get("issues", [])


def generate_runtime_flow(system, components, relationships, style):
    prompt = f"""
You are designing the runtime behavior of a software system.

System: {system}
Architecture Style: {style}

Components:
{json.dumps(components, indent=2)}

Relationships:
{json.dumps(relationships, indent=2)}

Describe the runtime interaction flow as JSON.

Return ONLY valid JSON in the following format:
{{
  "steps": [
    {{
      "from": "Component name",
      "to": "Component name",
      "action": "Meaningful action name",
      "mode": "sync | async"
    }}
  ]
}}
"""

    return robust_llm_json(prompt).get("steps", [])



def fallback_runtime_flow(style):
    style = style.lower()

    if "event" in style:
        return [
            {
                "from": "Order Service",
                "to": "Message Broker",
                "action": "Publish order created event",
                "mode": "async"
            },
            {
                "from": "Message Broker",
                "to": "Notification Service",
                "action": "Trigger user notification",
                "mode": "async"
            }
        ]

    # default layered
    return [
        {
            "from": "API",
            "to": "Service",
            "action": "Process business request",
            "mode": "sync"
        },
        {
            "from": "Service",
            "to": "Database",
            "action": "Persist data",
            "mode": "sync"
        }
    ]

# ================= ORCHESTRATOR =================

def ai_generate_architecture(system, frs, nfrs, style):
    try:
        decisions = extract_decisions(system, frs, nfrs, style)
        components = generate_components(system, frs)
        relationships = generate_relationships(components)
        steps = generate_runtime_flow(system, components, relationships, style)
        source = "AI"

    except Exception as e:
        print("⚠️ AI failed, using fallback:", e)
        components = fallback_components(style)
        relationships = fallback_relationships(style)
        steps = fallback_runtime_flow(style)   # 👈 ده المهم
        decisions = [{
            "name": "Fallback architecture",
            "rationale": "AI output invalid"
        }]
        source = "FALLBACK"

    return {
        "system": system,
        "style": style,
        "source": source,
        "decisions": decisions,
        "production_intents": style_production_intents(style),
        "components": components,
        "relationships": relationships,
        "runtime_flow": steps,
        "critique": critique(components, relationships, nfrs)
    }
