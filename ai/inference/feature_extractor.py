import json
import re
from huggingface_hub import InferenceClient
from ai.inference.feature_validator import validate_features
from ai.inference.prompt_builder import build_prompt
from ai.inference.response_parser import parse_response

# MODEL
client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct")


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
    text = re.sub(r"[^\w\s]", "", text)  # يشيل الرموز
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

    # 🔥 نعمل preprocessing هنا
    full_text = normalize_text(full_text)

    prompt = build_prompt(frs_text, nfrs_text)

    response = call_llm(prompt)

    print("RAW RESPONSE:\n", response)

    features = parse_response(response)

    # 👇 validation يستخدم النص النضيف
    features = validate_features(features, full_text)

    return features


def save_features(features):
    with open("data/outputs/features.json", "w") as f:
        json.dump(features, f, indent=4)


if __name__ == "__main__":
    features = extract_features()
    save_features(features)
    print(features)