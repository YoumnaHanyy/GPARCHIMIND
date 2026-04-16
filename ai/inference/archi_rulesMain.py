import json
from ai.inference.semantic_map import SEMANTIC_MAP
from ai.inference.engine import (
    apply_rules,
    apply_architecture,
    apply_negative_rules,
    normalize_scores,
    get_top_patterns,
    explain_scores,
    classify_confidence
)


# =========================================
# 🔥 SEMANTIC BOOST (NEW)
# =========================================
def semantic_boost(feature, evidence):
    config = SEMANTIC_MAP.get(feature, {})

    # safeguard
    if not isinstance(config, dict):
        return 0

    score = 0

    for text in evidence:
        text = text.lower()

        # 🔥 strong matches
        for kw in config.get("strong", []):
            if kw in text:
                score += 0.3

        # 🔥 medium matches
        for kw in config.get("medium", []):
            if kw in text:
                score += 0.2

        # 🔥 weak matches
        for kw in config.get("weak", []):
            if kw in text:
                score += 0.1

        # 🔥 tech keywords
        for kw in config.get("tech", []):
            if kw in text:
                score += 0.25

    return min(score, 0.5)


# =========================================
# 🔹 تحويل raw features → numeric
# =========================================
def extract_numeric_features(raw_features):
    numeric_features = {}

    for key, value in raw_features.items():
        supported = value.get("supported")
        confidence = value.get("confidence", 0)
        evidence = value.get("evidence", [])

        score = 0

        # 🔹 base score
        if supported is True:
            score = confidence
        elif supported == "partial":
            score = confidence * 0.75

        # 🔥 NEW: semantic boost
        boost = semantic_boost(key, evidence)
        score = min(1.0, score + boost)

        numeric_features[key] = min(score, 1.0)

    return numeric_features


# =========================================
# 🎯 MAIN INFERENCE
# =========================================
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

    # 🔹 convert features
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


# =========================================
# 🧪 TEST / STANDALONE
# =========================================
def run_design_patterns(input_data):
    output = run_inference(input_data)

    print("\n=== Top Design Pattern Recommendations ===")

    for item in output["top_patterns"]:
        print(f"\n{item['pattern']} → {item['score']} ({item['confidence']})")

        for reason in item["reasons"]:
            print("  -", reason)

    with open("data/outputs/pattern_results.json", "w") as f:
        json.dump(output, f, indent=4)

    print("\n✅ Results saved to pattern_results.json")

    return output