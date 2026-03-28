import json
import re
from huggingface_hub import InferenceClient
from infrastructure.database import db
from ai.inference.feature_validator import validate_features
from ai.inference.prompt_builder import build_prompt
from ai.inference.response_parser import parse_response
from ai.utils.feature_keywords import FEATURE_KEYWORDS
# MODEL
client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct")

ALL_FEATURES = [
    "EVENT_DRIVEN",
    "REAL_TIME",
    "HIGH_SCALABILITY",
    "LOW_LATENCY",
    "HIGH_EXTENSIBILITY",
    "HIGH_MAINTAINABILITY",
    "FLEXIBLE_CREATION",
    "DYNAMIC_BEHAVIOR",
    "DISTRIBUTED_SYSTEM",
    "FAULT_TOLERANCE",
    "HIGH_SECURITY",
    "MODULARITY",
    "HIGH_COUPLING_RISK"
]

def ensure_all_features(features):
    for key in ALL_FEATURES:
        if key not in features:
            features[key] = 0.0
    return features

def load_requirements():
    with open("data/outputs/functional_requirements.json", "r") as f:
        frs = json.load(f)

    with open("data/outputs/non_functional_requirements.json", "r") as f:
        nfrs = json.load(f)

    return frs, nfrs



def format_requirements(frs, nfrs):

    def extract_text(items):
        result = []
        
        for item in items:
            if isinstance(item, dict):
                title = item.get("title", "")
                desc = item.get("description", "")
                result.append(f"{title}: {desc}")
            else:
                result.append(str(item))
        
        return "\n".join(result)

    frs_text = extract_text(frs)
    nfrs_text = extract_text(nfrs)

    return frs_text, nfrs_text



def normalize_text(text):
    text = text.lower()
    text = text.replace("-", " ")
    return text



def call_llm(prompt):
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.2
    )

    return response.choices[0].message.content

def extract_features():
    frs, nfrs = load_requirements()

    frs_text, nfrs_text = format_requirements(frs, nfrs)

    full_text = frs_text + "\n" + nfrs_text

    prompt = build_prompt(frs_text, nfrs_text)

    # 🔁 MULTI-RUN (المهم)
    runs = []

    for _ in range(3):
        response = call_llm(prompt)
        parsed = parse_response(response)
        runs.append(parsed)

    # 🧠 average
    ai_features = {}

    for k in runs[0]:
        ai_features[k] = sum(r.get(k, 0.0) for r in runs) / len(runs)

    # validation
    validated = validate_features(ai_features.copy(), full_text)

    # merge
    features = {}

    for k in set(ai_features) | set(validated):
        features[k] = max(
            ai_features.get(k, 0.0),
            validated.get(k, 0.0)
        )

    features = ensure_all_features(features)

    return features

def match_keyword(word, text):
    return (
        word in text or
        word.replace(" ", "") in text or
        word.replace("-", " ") in text or
        any(w in text for w in word.split())
    )


def extract_evidence(feature, text):
    evidence = []

    for word in FEATURE_KEYWORDS.get(feature, []):
        if match_keyword(word, text):
            evidence.append(word)

    return list(set(evidence))
def build_feature_decision(features, text):
    decisions = {}
    text = normalize_text(text)

    for feature, score in features.items():

        if score >= 0.7:
            supported = True
        elif score <= 0.3:
            supported = False
        else:
            supported = "partial"

        evidence = extract_evidence(feature, text)

        # fallback
        if not evidence and score > 0.5:
            evidence = ["derived from context"]

        decisions[feature] = {
            "supported": supported,
            "confidence": round(score, 2),
            "evidence": evidence
        }

    return decisions
def save_features(features):
    with open("data/outputs/features.json", "w") as f:
        json.dump(features, f, indent=4)

def load_selected_architecture(project_id):

    hybrid_doc = db.hybrid_method.find_one({"project_id": project_id})

    if not hybrid_doc or not hybrid_doc.get("selected_architecture"):
        raise ValueError("No selected architecture found")

    return hybrid_doc["selected_architecture"]



def get_latest_project_id():
    doc = db.hybrid_method.find_one(
        {"selected_architecture": {"$exists": True}},
        sort=[("_id", -1)]
    )

    if not doc:
        raise ValueError("No project with selected architecture found")

    return doc["project_id"]
def save_decisions(decisions, architecture="LAYERED"):
    output = {
        "architecture": architecture,
        "features": decisions
    }

    with open("data/outputs/feature_decisions.json", "w") as f:
        json.dump(output, f, indent=4)



if __name__ == "__main__":
    features = extract_features()

    frs, nfrs = load_requirements()
    frs_text, nfrs_text = format_requirements(frs, nfrs)
    full_text = frs_text + "\n" + nfrs_text

    decisions = build_feature_decision(features, full_text)

    # 🔥 automatic بدل ما تكتبيه
    project_id = get_latest_project_id()
    architecture = load_selected_architecture(project_id)

    save_decisions(decisions, architecture)

    print("✅ Using project:", project_id)
    print(decisions)