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

Extract:
1. Actors
2. Use cases (SHORT: 2-4 words)
3. Map each use case to actor

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

    return "\n".join(lines),data