from huggingface_hub import InferenceClient
import json

# ================= LLM =================

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

def robust_llm_json(prompt, retries=3):
    for _ in range(retries):
        try:
            return extract_json(ask_llm(prompt))
        except Exception:
            prompt = f"""
RETURN ONLY VALID JSON.
NO text. NO markdown.

TASK:
{prompt}
"""
    raise ValueError("LLM failed to return valid JSON")

# ================= STYLE-AWARE RULES =================

def style_production_intents(style: str):
    style = style.lower()

    if "event" in style:
        return {
            "deployment": {
                "scaling": "horizontal",
                "availability": "multi-region"
            },
            "security": {
                "authentication": "service-to-service",
                "authorization": "RBAC",
                "encryption": ["in-transit"]
            },
            "resilience": {
                "patterns": ["Retry", "CircuitBreaker"],
                "delivery": "at-least-once",
                "ordering": False
            }
        }

    if "pipe" in style or "pipeline" in style:
        return {
            "deployment": {
                "scaling": "horizontal",
                "availability": "single-region"
            },
            "security": {
                "authentication": "internal",
                "authorization": "RBAC"
            },
            "resilience": {
                "patterns": ["Retry"],
                "delivery": "exactly-once",
                "ordering": True
            }
        }

    if "layered" in style:
        return {
            "deployment": {
                "scaling": "vertical",
                "availability": "single-region"
            },
            "security": {
                "authentication": "centralized",
                "authorization": "RBAC"
            },
            "resilience": {
                "patterns": ["Retry"],
                "delivery": "at-most-once",
                "ordering": True
            }
        }

    # default (microservices)
    return {
        "deployment": {
            "scaling": "horizontal",
            "availability": "multi-region"
        },
        "security": {
            "authentication": "service-to-service",
            "authorization": "RBAC"
        },
        "resilience": {
            "patterns": ["Retry", "CircuitBreaker"],
            "delivery": "at-least-once",
            "ordering": False
        }
    }

# ================= STEPS =================

def extract_decisions(system, frs, nfrs, style):
    prompt = f"""
System: {system}
Architecture Style: {style}

Functional Requirements:
{frs}

Non-Functional Requirements:
{nfrs}

Return JSON:
{{
  "decisions": [
    {{ "name": "...", "rationale": "..." }}
  ]
}}
"""
    return robust_llm_json(prompt).get("decisions", [])

def generate_components(system, frs):
    prompt = f"""
System: {system}

Functional Requirements:
{frs}

Return JSON:
{{
  "components": [
    {{ "name": "...", "responsibility": "..." }}
  ]
}}
"""
    return robust_llm_json(prompt)["components"]

def generate_relationships(components):
    prompt = f"""
Components:
{json.dumps(components, indent=2)}

Return JSON:
{{
  "relationships": [
    {{ "source": "...", "target": "...", "type": "data-flow | event-flow" }}
  ]
}}
"""
    return robust_llm_json(prompt)["relationships"]

def critique(components, relationships, nfrs):
    prompt = f"""
Components:
{components}

Relationships:
{relationships}

NFRs:
{nfrs}

Return JSON:
{{ 
  "issues": []

 }}
"""
    return robust_llm_json(prompt).get("issues", [])

# ================= ORCHESTRATOR =================

def ai_generate_architecture(system, frs, nfrs, style):
    decisions = extract_decisions(system, frs, nfrs, style)
    components = generate_components(system, frs)
    relationships = generate_relationships(components)

    return {
        "decisions": decisions,
        "production_intents": style_production_intents(style),
        "components": components,
        "relationships": relationships,
        "critique": critique(components, relationships, nfrs)
    }
