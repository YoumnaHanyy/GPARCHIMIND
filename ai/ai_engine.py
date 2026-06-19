from dotenv import load_dotenv
load_dotenv(override=True)
import os
print(os.getenv("HF_API_KEY"))
import json
import re
from huggingface_hub import InferenceClient


HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY not found in environment variables")

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

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
    print("RESPONSE TYPE:", type(response))
    print(response)

    return response.choices[0].message.content


# ================= SAFE JSON EXTRACTION =================

def extract_json(text: str):

    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1)

    if text.startswith("```"):
        text = text.replace("```", "", 1)

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        return json.loads(text)

    except:
        pass

    try:

        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end <= 0:
            raise ValueError("No JSON object found")

        json_str = text[start:end]

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
            print("\n========== RAW LLM ==========")
            print(response)
            print("=============================\n")
            print("\n========== PROMPT ==========")
            print(prompt[:2000])
            print("============================\n")
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


    print("STEP: extract_decisions")

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
    print("\nDECISIONS PROMPT:")
    print(prompt[:1000])

    result = robust_llm_json(prompt)

    print("\nDECISIONS RESULT:")
    print(type(result))
    print(result)

    if isinstance(result, list):
     print("WARNING: LLM returned LIST instead of JSON OBJECT")
     return []

    return result.get("decisions", [])


def generate_components(system, frs , style):
    print("STEP: generate_components")

    prompt = f"""
System: {system}
Architecture Style: {style}


Functional Requirements:
{frs}
Generate components suitable for the selected architecture style.
Choose component kinds and layers according to the selected architecture style.


IMPORTANT:
- Limit to maximum 8 components
- Keep output short
- Every component must have:
  - name
  - responsibility
  - kind

STRICT RULES:
- RETURN JSON ONLY
- NO explanation
- NO markdown

The generated components MUST strictly match the selected architecture style.

Do NOT generate components from other architecture styles.

Examples:
- MVC must not contain Broker.
- Broker must not contain Controller.
- Layered must not contain Event Bus.
- Hexagonal must not contain MVC components.

If Architecture Style is Monolithic:
generate UI, Business Logic and Database components inside one application.

If Architecture Style is Layered:
generate Presentation, Business and Data components.

If Architecture Style is MVC:
generate Model, View and Controller components.

If Architecture Style is Client-Server:
generate clients, servers and databases.

If Architecture Style is Microservices:
generate independent services, databases and API Gateway.

If Architecture Style is Service-Oriented Architecture:
generate service providers, service consumers, shared services and service registry.

If Architecture Style is Event-Driven:
generate producers, consumers and event brokers.

If Architecture Style is Microkernel:
generate core system, plugins and extension points.

If Architecture Style is Component-Based:
generate reusable components and connectors.

If Architecture Style is Pipe-and-Filter:
generate filters and pipes.

If Architecture Style is Broker:
generate broker, clients and servers.

If Architecture Style is Peer-to-Peer:
generate peer nodes with decentralized communication.

If Architecture Style is Blackboard:
generate blackboard repository, knowledge sources and controller.

If Architecture Style is Space-Based:
generate processing units, data grid and messaging infrastructure.

If Architecture Style is REST:
generate REST APIs, resources and clients.

If Architecture Style is Hexagonal:
generate application core, ports and adapters.

If Architecture Style is Serverless:
generate functions, event triggers and managed cloud services.

If Architecture Style is Event-Bus:
generate publishers, subscribers and event bus.

{{
 "components":[
  {{
   "name":"Component name",
   "responsibility":"Short responsibility",
   "kind":"service | database | external | broker | gateway |ui | controller | model | view | adapter | port | filter | pipe | client | server | plugin | core | registry | bus | publisher | subscriber | knowledge-source | blackboard | processing-unit | datastore",
    "technology":"Technology name"
  
  }}
 ]
}}
"""

    data = robust_llm_json(prompt)

    if isinstance(data, list):
     return data

    return data.get("components", [])


def generate_relationships(components, style):

    prompt = f"""
    Architecture Style: {style}

Components:

{json.dumps(components, indent=2)}

IMPORTANT:
- Keep it concise

STRICT RULES:
- RETURN JSON ONLY
- NO explanation
- NO markdown

The generated relationships MUST strictly follow the selected architecture style.
Do NOT create relationships that violate the architecture style.
Use ONLY the provided components.

Rules:
If Architecture Style is Monolithic:
all modules communicate directly within the same application.

If Architecture Style is Layered:
Presentation -> Business -> Data.

If Architecture Style is MVC:
Controller interacts with Model and View.

If Architecture Style is Client-Server:
clients communicate with servers which access databases.

If Architecture Style is Microservices:
services communicate through APIs and databases.

If Architecture Style is Service-Oriented Architecture:
service consumers communicate with service providers through service contracts.

If Architecture Style is Event-Driven:
producers publish events to broker and consumers subscribe.

If Architecture Style is Microkernel:
plugins communicate with the core system.

If Architecture Style is Component-Based:
components communicate through connectors and interfaces.

If Architecture Style is Pipe-and-Filter:
filters communicate through pipes.

If Architecture Style is Broker:
clients communicate with servers through broker.

If Architecture Style is Peer-to-Peer:
peers communicate directly with other peers.

If Architecture Style is Blackboard:
knowledge sources read and write to blackboard repository.

If Architecture Style is Space-Based:
processing units communicate through distributed data grid.

If Architecture Style is REST:
clients communicate with REST resources through HTTP APIs.

If Architecture Style is Hexagonal:
adapters communicate through ports to application core.

If Architecture Style is Serverless:
functions communicate through events and managed services.

If Architecture Style is Event-Bus:
publishers publish to event bus and subscribers consume events.


{{
 "relationships":[
  {{
   "source":"Component",
   "target":"Component",
   "type":"data-flow | event-flow",
   "description":"Short interaction description"
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
- Max 8 steps
- Keep short

STRICT RULES:
- RETURN JSON ONLY
- NO explanation
- NO markdown

The generated runtime flow MUST strictly follow the selected architecture style.
Do NOT invent components.
Use ONLY the components provided above.

Rules:
If Architecture Style is Monolithic:
user -> ui -> business logic -> database

If Architecture Style is Layered:
presentation -> business -> data -> database

If Architecture Style is MVC:
user -> controller -> model -> view

If Architecture Style is Client-Server:
client request -> server -> database -> client response

If Architecture Style is Microservices:
client -> api gateway -> service -> database

If Architecture Style is Service-Oriented Architecture:
consumer -> service registry -> provider -> response

If Architecture Style is Event-Driven:
producer -> broker -> consumer

If Architecture Style is Microkernel:
plugin -> core system -> plugin response

If Architecture Style is Component-Based:
component -> connector -> component

If Architecture Style is Pipe-and-Filter:
filter -> pipe -> filter

If Architecture Style is Broker:
client request -> broker -> server response

If Architecture Style is Peer-to-Peer:
peer -> peer -> peer

If Architecture Style is Blackboard:
knowledge source -> blackboard -> knowledge source

If Architecture Style is Space-Based:
client -> processing unit -> data grid

If Architecture Style is REST:
client -> rest api -> resource -> response

If Architecture Style is Hexagonal:
adapter -> port -> application core -> port -> adapter

If Architecture Style is Serverless:
event trigger -> function -> managed service

If Architecture Style is Event-Bus:
publisher -> event bus -> subscriber

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

Return at least 3 architecture observations.
Include strengths and weaknesses.
Return ONLY valid JSON:

{{
 "issues":["Observation"]
}}
"""

    try:

        data = robust_llm_json(prompt)

        if isinstance(data, list):
            return data

        return data.get("issues", [])

    except Exception as e:

        print("Critique generation failed:", e)

        return [
            "No critique available"
        ]


# ================= ORCHESTRATOR =================

def ai_generate_architecture(system, frs, nfrs, style):

    try:

        decisions = extract_decisions(system, frs, nfrs, style)

        components = generate_components(system, frs, style)

        relationships = generate_relationships(components, style)

        steps = generate_runtime_flow(
            system,
            components,
            relationships,
            style
        )

        # critique داخل الـ try
        critique_result = critique(
            components,
            relationships,
            nfrs
        )

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

        critique_result = [
            "Architecture generated using fallback mode"
        ]

        source = "FALLBACK"
    print("\n===================")
    print("SOURCE =", source)
    print("COMPONENTS =", components)
    print("===================\n")    

    return {
        "system": system,
        "style": style,
        "source": source,
        "decisions": decisions,
        "production_intents": style_production_intents(style),
        "components": components,
        "relationships": relationships,
        "runtime_flow": steps,
        "critique": critique_result
    }
def test_llm():
    response = ask_llm("""
Return ONLY this JSON:

{
  "test": "hello"
}
""")

    print(response)

if __name__ == "__main__":
    test_llm()