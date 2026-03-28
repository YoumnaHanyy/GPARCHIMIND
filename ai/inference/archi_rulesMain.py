import json
from semantic_map import SEMANTIC_MAP
from engine import (
    apply_rules,
    apply_architecture,
    apply_negative_rules,
    normalize_scores,
    get_top_patterns,
    explain_scores,
    classify_confidence
)


# 🔹 تحويل الـ raw features → numeric features
def extract_numeric_features(raw_features):
    numeric_features = {}

    for key, value in raw_features.items():
        supported = value.get("supported")
        confidence = value.get("confidence", 0)
        evidence = value.get("evidence", [])

        score = 0

        # 🔹 base score من supported/confidence
        if supported is True:
            score = confidence
        elif supported == "partial":
            score = confidence * 0.5

        # 🔥 semantic matching
        keywords = SEMANTIC_MAP.get(key, [])
        semantic_hit = False

        for text in evidence:
            text = text.lower()

            for kw in keywords:
                if kw in text:
                    semantic_hit = True
                    break

            if semantic_hit:
                break

        # 🔥 لو semantic hit → نرفع القيمة
        if semantic_hit:
            score = max(score, 0.6)

        numeric_features[key] = min(score, 1.0)

    return numeric_features


# 🎯 الدالة الأساسية اللي هتستخدميها في المشروع الكبير
def run_inference(input_data: dict):
    """
    Takes input JSON (dict) and returns:
    - top patterns
    - full scores
    - explanations
    """

    architecture = input_data.get("architecture")

    if not architecture:
        raise ValueError("Architecture is missing in input!")

    raw_features = input_data.get("features", input_data)

    # 🔹 تحويل features
    features = extract_numeric_features(raw_features)

    # 🔹 pipeline
    scores = apply_rules(features)
    scores = apply_negative_rules(scores, features)
    scores = apply_architecture(scores, architecture)
    scores = normalize_scores(scores)

    top = get_top_patterns(scores)
    explanations = explain_scores(features, scores)

    # 🔹 format output
    results = []

    for pattern, score in top:
        results.append({
            "pattern": pattern,
            "score": score,
            "confidence": classify_confidence(score),
            "reasons": explanations.get(pattern, [])
        })

    return {
        "top_patterns": results,
        "all_scores": scores
    }


# 🧪 optional: لو حابة تشغليه standalone للتست
if __name__ == "__main__":
    with open("data/outputs/feature_decisions.json", "r") as f:
        data = json.load(f)

    output = run_inference(data)

  # 🔹 print
    print("\n=== Top Design Pattern Recommendations ===")

    for item in output["top_patterns"]:
     print(f"\n{item['pattern']} → {item['score']} ({item['confidence']})")

    for reason in item["reasons"]:
        print("  -", reason)

# 🔹 save
    with open("data/outputs/pattern_results.json", "w") as f:
     json.dump(output, f, indent=4)

    print("\n✅ Results saved to pattern_results.json")