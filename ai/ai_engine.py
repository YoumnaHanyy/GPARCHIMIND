import os
import json
import re
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY not found in environment variables")

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

client = InferenceClient(
    model=MODEL_NAME,
    token=HF_API_KEY
)

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


# ================= SAFE JSON EXTRACTION =================

def extract_json(text: str):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            raise ValueError("No JSON object found")

        json_str = match.group()

        # remove trailing commas
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        return json.loads(json_str)

    except Exception as e:
        raise ValueError(f"Invalid JSON returned by LLM: {e}")


# ================= ROBUST LLM JSON =================

def robust_llm_json(prompt, retries=4):

    last_error = None

    for _ in range(retries):
        try:
            response = ask_llm(prompt)
            return extract_json(response)

        except Exception as e:
            last_error = e
            prompt = f"""
RETURN ONLY VALID JSON.
NO explanations.
NO markdown.
NO comments.

{prompt}
"""

    raise RuntimeError(f"LLM failed after {retries} attempts: {last_error}")


# ================= FALLBACK COMPONENTS =================

def fallback_components(style):

    style = style.lower()

    if "event" in style:
        return [
            {"name": "Order Service", "responsibility": "Manage orders"},
            {"name": "Payment Service", "responsibility": "Process payments"},
            {"name": "Notification Service", "responsibility": "Send notifications"},
            {"name": "Message Broker", "responsibility": "Async communication"},
            {"name": "Order Database", "responsibility": "Persist order data"}
        ]

    if "micro" in style:
        return [
            {"name": "API Gateway", "responsibility": "Route requests"},
            {"name": "User Service", "responsibility": "Manage users"},
            {"name": "Order Service", "responsibility": "Manage orders"},
            {"name": "Order Database", "responsibility": "Persist orders"}
        ]

    return [
        {"name": "API", "responsibility": "Handle requests"},
        {"name": "Service", "responsibility": "Business logic"},
        {"name": "Database", "responsibility": "Persistent storage"}
    ]


# ================= FALLBACK RELATIONSHIPS =================

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

    return [
        {"source": "API", "target": "Service", "type": "data-flow"},
        {"source": "Service", "target": "Database", "type": "data-flow"}
    ]


# ================= RUNTIME FALLBACK =================

def fallback_runtime_flow(style):

    style = style.lower()

    if "event" in style:
        return [
            {
                "from": "Order Service",
                "to": "Message Broker",
                "action": "Publish order event",
                "mode": "async"
            },
            {
                "from": "Message Broker",
                "to": "Notification Service",
                "action": "Send notification",
                "mode": "async"
            }
        ]

    return [
        {
            "from": "API",
            "to": "Service",
            "action": "Process request",
            "mode": "sync"
        },
        {
            "from": "Service",
            "to": "Database",
            "action": "Persist data",
            "mode": "sync"
        }
    ]


# ================= STYLE PRODUCTION INTENTS =================

def style_production_intents(style: str):

    style = style.lower()

    if "event" in style:
        return {
            "deployment": {"scaling": "horizontal", "availability": "multi-region"},
            "security": {"authentication": "service-to-service", "authorization": "RBAC"},
            "resilience": {"patterns": ["Retry", "CircuitBreaker"], "delivery": "at-least-once"}
        }

    if "layered" in style:
        return {
            "deployment": {"scaling": "vertical", "availability": "single-region"},
            "security": {"authentication": "centralized", "authorization": "RBAC"},
            "resilience": {"patterns": ["Retry"], "delivery": "at-most-once"}
        }

    return {
        "deployment": {"scaling": "horizontal", "availability": "multi-region"},
        "security": {"authentication": "service-to-service", "authorization": "RBAC"},
        "resilience": {"patterns": ["Retry", "CircuitBreaker"], "delivery": "at-least-once"}
    }


# ================= LLM TASKS =================

def extract_decisions(system, frs, nfrs, style):

    prompt = f"""
System: {system}
Architecture Style: {style}

Functional Requirements:
{frs}

Non Functional Requirements:
{nfrs}

Return ONLY valid JSON:

{{
 "decisions":[
  {{
   "name":"Decision name",
   "rationale":"Short explanation"
  }}
 ]
}}
"""

    return robust_llm_json(prompt).get("decisions", [])


def generate_components(system, frs):

    prompt = f"""
System: {system}

Functional Requirements:
{frs}

Return ONLY valid JSON:

{{
 "components":[
  {{
   "name":"Component name",
   "responsibility":"Short responsibility"
  }}
 ]
}}
"""

    return robust_llm_json(prompt).get("components", [])


def generate_relationships(components):

    prompt = f"""
Components:

{json.dumps(components, indent=2)}

Return ONLY valid JSON:

{{
 "relationships":[
  {{
   "source":"Component",
   "target":"Component",
   "type":"data-flow | event-flow"
  }}
 ]
}}
"""

    return robust_llm_json(prompt).get("relationships", [])


def generate_runtime_flow(system, components, relationships, style):

    prompt = f"""
System: {system}
Architecture Style: {style}

Components:
{json.dumps(components, indent=2)}

Relationships:
{json.dumps(relationships, indent=2)}

Return ONLY valid JSON:

{{
 "steps":[
  {{
   "from":"Component",
   "to":"Component",
   "action":"Action description",
   "mode":"sync | async"
  }}
 ]
}}
"""

    return robust_llm_json(prompt).get("steps", [])


def critique(components, relationships, nfrs):

    prompt = f"""
Components:
{json.dumps(components, indent=2)}

Relationships:
{json.dumps(relationships, indent=2)}

NFRs:
{nfrs}

Return ONLY valid JSON:

{{
 "issues":[]
}}
"""

    return robust_llm_json(prompt).get("issues", [])


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

        steps = fallback_runtime_flow(style)

        decisions = [
            {
                "name": "Fallback Architecture",
                "rationale": "AI response invalid"
            }
        ]

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