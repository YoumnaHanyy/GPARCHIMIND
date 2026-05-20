import os
import json
import re
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

logger = logging.getLogger("ai_engine")

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
            {"role": "system", "content": "You are a strict JSON generator. Return ONLY raw JSON. No explanation, no markdown, no extra text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=temperature
    )

    return response["choices"][0]["message"]["content"]


# ================= SAFE JSON EXTRACTION =================

def extract_json(text: str):
    try:
        # نحاول direct parse
        return json.loads(text)

    except:
        pass

    try:
        # نلقط أول JSON block مظبوط
        start = text.find("{")
        end = text.rfind("}") + 1
        json_str = text[start:end]

        # تنظيف
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        return json.loads(json_str)

    except Exception as e:
        raise ValueError(f"Invalid JSON returned by LLM: {e}")
# ================= ROBUST LLM JSON =================

def robust_llm_json(prompt, retries=4):

    last_error = None
    original_prompt = prompt

    for i in range(retries):
        try:
            response = ask_llm(prompt)
            return extract_json(response)

        except Exception as e:
            last_error = e

            # 🔥 FIX: self-healing بدل ما يعيد نفس الغلط
            prompt = f"""
The previous response was INVALID JSON.

Error:
{last_error}

Fix it and return ONLY valid JSON.

STRICT RULES:
- NO explanation
- NO markdown
- NO text before or after
- ONLY RAW JSON

Original request:
{original_prompt}
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

IMPORTANT:
- Limit to maximum 8 components
- Keep output short

STRICT RULES:
- RETURN JSON ONLY
- NO explanation
- NO markdown

{{
 "components":[
  {{
   "name":"Component name",
   "responsibility":"Short responsibility"
  }}
 ]
}}
"""

    data = robust_llm_json(prompt)

    if isinstance(data, list):
     return data

    return data.get("components", [])


def generate_relationships(components):

    prompt = f"""
Components:

{json.dumps(components, indent=2)}

IMPORTANT:
- Keep it concise

STRICT RULES:
- RETURN JSON ONLY
- NO explanation
- NO markdown

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

    data = robust_llm_json(prompt)

    if isinstance(data, list):
     return data

    return data.get("relationships", [])


def generate_runtime_flow(system, components, relationships, style):

    prompt = f"""
System: {system}
Architecture Style: {style}

Components:
{json.dumps(components, indent=2)}

Relationships:
{json.dumps(relationships, indent=2)}

IMPORTANT:
- Max 6 steps
- Keep short

STRICT RULES:
- RETURN JSON ONLY
- NO explanation
- NO markdown

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

    data = robust_llm_json(prompt)

    if isinstance(data, list):
     return data

    return data.get("steps", [])


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

def ai_generate_architecture(system, frs, nfrs, style, enable_critique=False):

    t_start = time.time()
    logger.info("[ai_generate_architecture] START system=%s style=%s", system, style)

    source = "AI"

    # Stage 1: decisions + components are independent — run in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        t1 = time.time()
        fut_decisions  = pool.submit(extract_decisions, system, frs, nfrs, style)
        fut_components = pool.submit(generate_components, system, frs)

        try:
            decisions = fut_decisions.result()
            logger.info("[extract_decisions] done %.2fs", time.time() - t1)
        except Exception as e:
            logger.warning("[extract_decisions] failed %.2fs: %s", time.time() - t1, e)
            decisions = [{"name": "Fallback Architecture", "rationale": "AI response invalid"}]

        try:
            components = fut_components.result()
            logger.info("[generate_components] done %.2fs", time.time() - t1)
        except Exception as e:
            logger.warning("[generate_components] failed %.2fs: %s", time.time() - t1, e)
            components = fallback_components(style)
            source = "FALLBACK"

    # Stage 2: relationships needs components
    t2 = time.time()
    try:
        relationships = generate_relationships(components)
        logger.info("[generate_relationships] done %.2fs", time.time() - t2)
    except Exception as e:
        logger.warning("[generate_relationships] failed %.2fs: %s", time.time() - t2, e)
        relationships = fallback_relationships(style)
        source = "FALLBACK"

    # Stage 3: runtime_flow + (optional) critique both need components + relationships — run in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        t3 = time.time()
        fut_runtime  = pool.submit(generate_runtime_flow, system, components, relationships, style)
        fut_critique = pool.submit(critique, components, relationships, nfrs) if enable_critique else None

        try:
            steps = fut_runtime.result()
            logger.info("[generate_runtime_flow] done %.2fs", time.time() - t3)
        except Exception as e:
            logger.warning("[generate_runtime_flow] failed %.2fs: %s", time.time() - t3, e)
            steps = fallback_runtime_flow(style)
            source = "FALLBACK"

        critique_issues = []
        if fut_critique is not None:
            try:
                critique_issues = fut_critique.result()
                logger.info("[critique] done %.2fs", time.time() - t3)
            except Exception as e:
                logger.warning("[critique] failed (non-critical) %.2fs: %s", time.time() - t3, e)

    t_end = time.time()
    logger.info("[ai_generate_architecture] END source=%s elapsed=%.2fs", source, t_end - t_start)

    return {
        "system": system,
        "style": style,
        "source": source,
        "decisions": decisions,
        "production_intents": style_production_intents(style),
        "components": components,
        "relationships": relationships,
        "runtime_flow": steps,
        "critique": critique_issues,
    }