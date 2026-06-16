from ai.ai_engine import ask_llm, extract_json
import re


# =========================
# 🧠 NORMALIZE AI OUTPUT
# =========================
def normalize_ai_output(data):
    actors = data.get("actors") or data.get("actor") or []
    usecases = data.get("usecases") or data.get("use_cases") or []

    # ensure structure صح
    fixed_usecases = []
    for uc in usecases:
        if isinstance(uc, dict):
            name = uc.get("name") or uc.get("usecase") or ""
            actor = uc.get("actor") or uc.get("user") or "User"

            if name:
                fixed_usecases.append({
                    "name": name.strip(),
                    "actor": actor.strip()
                })

    return {
        "actors": list(set(actors)),
        "usecases": fixed_usecases
    }


# =========================
# 💣 MAIN FUNCTION
# =========================
def generate_usecase_ai(frs, system_name):

    # 🧠 حضّري النص
    frs_text = "\n".join([
        f"- {fr.get('description', '')}" for fr in frs
    ])

    prompt = f"""
You are a senior software architect.

Extract realistic actors from the requirements.

STRICT RULES:
- Do NOT use only "User".
- Infer domain-specific actors when possible.
- Examples:
  CV System -> Job Seeker, Recruiter, Admin
  E-Commerce -> Customer, Seller, Admin
  Hospital -> Patient, Doctor, Receptionist
  Banking -> Customer, Bank Employee, Admin



  

You are a senior software architect.

Generate a UML Use Case Model from the functional requirements.

STRICT RULES:

- A Use Case must represent a goal achieved by an external actor.
- Every Use Case must be directly traceable to one or more functional requirements.
- Infer realistic domain-specific actors from the requirements.
- Do NOT use generic actors unless no other actor can be inferred.
- Use concise names (2-4 words maximum).
- Use verb + object naming style.

DO NOT INCLUDE:
- Internal algorithms
- Technical implementations
- AI/NLP processing
- Database operations
- Calculations
- Background system activities
- Internal services or components
- System-to-system internal behavior

A use case must answer:
"What does an external user/stakeholder want to accomplish?"

Examples of valid use cases:
- Register Account
- Submit Application
- Approve Request
- Generate Report
- View Dashboard
- Manage Inventory

Examples of invalid use cases:
- Process Data
- Execute Algorithm
- Calculate Similarity
- Store Records
- Query Database
- Run AI Model

For each use case:
- Assign the most appropriate actor.
- Only include actors that actually participate in the system.

Return ONLY valid JSON.

FORMAT:
{{
  "actors": [
    "Actor 1",
    "Actor 2"
  ],
  "usecases": [
    {{
      "name": "Use Case Name",
      "actor": "Actor 1"
    }}
  ]
}}

STRICT RULES:
- No "shall"
- No long sentences
- Use verb + object (e.g., "Login", "Submit CV")

RETURN ONLY VALID JSON.
NO TEXT BEFORE OR AFTER.

FORMAT:
{{
  "actors": ["User", "Admin"],
  "usecases": [
    {{"name": "Login", "actor": "User"}}
  ]
}}

Requirements:
{frs_text}
"""

    # 🔥 AI call
    response = ask_llm(prompt)

    # 🧠 parse + fix
    data = extract_json(response)
    data = normalize_ai_output(data)
    if len(data["actors"]) == 1 and data["actors"][0].lower() == "user":
       data["actors"] = [
           "Primary User",
           "Administrator"
        ]

    # 🧪 debug (مهم جدًا دلوقتي)
    print("🔥 AI USECASE OUTPUT:", data)

    # =========================
    # 🎨 PlantUML
    # =========================

    def alias(x):
        return re.sub(r'[^a-zA-Z0-9_]', '', x)

    lines = []
    lines.append("@startuml")
    lines.append("left to right direction")
    lines.append("skinparam dpi 300")

    actors = data.get("actors", [])
    usecases = data.get("usecases", [])

    # ⚠️ safety fallback
    if not actors:
        actors = ["User"]

    if not usecases:
        usecases = [{"name": "Use System", "actor": "User"}]

    # 👤 actors
    for actor in actors:
        lines.append(f'actor "{actor}" as {alias(actor)}')

    lines.append(f'rectangle "{system_name}" {{')

    # 🎯 use cases
    for i, uc in enumerate(usecases):
        name = uc["name"]
        lines.append(f'  usecase "{name}" as UC{i}')

    lines.append("}")

    # 🔗 relations
    for i, uc in enumerate(usecases):
        actor_alias = alias(uc["actor"])
        lines.append(f'{actor_alias} --> UC{i}')

    lines.append("@enduml")
    import json

    print("\n========== USECASE DATA ==========")
    print(json.dumps(data, indent=2))
    print("==================================\n")
    

    return "\n".join(lines),data