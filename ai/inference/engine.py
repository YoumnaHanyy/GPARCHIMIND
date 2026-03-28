from rules import RULES
from architecture_rules import ARCH_RULES


def apply_rules(features):
    scores = {}

    for pattern, categories in RULES.items():
        score = 0

        core_count = 0
        generic_count = 0

        core_features = categories.get("core", [])
        generic_features = categories.get("generic", [])

        # 🔥 Core features
        for feature in core_features:
            value = features.get(feature, 0)
            if value > 0:
                score += value * 1.5
                core_count += 1

        # 🟡 Generic features
        for feature in generic_features:
            value = features.get(feature, 0)
            if value > 0:
                score += value * 0.5
                generic_count += 1

        total_matches = core_count + generic_count

        # 🔥 FINAL STABLE LOGIC (بدل penalties القديمة)
        if core_count >= 1:
            scores[pattern] = score

        elif generic_count >= 2:
            scores[pattern] = score * 0.5

        else:
            scores[pattern] = score * 0.2

    return scores


def apply_negative_rules(scores, features):
    if features.get("HIGH_COUPLING_RISK", 0) > 0.5:
        for pattern, categories in RULES.items():
            core_features = categories.get("core", [])
            generic_features = categories.get("generic", [])

            all_features = core_features + generic_features

            if "LOW_COUPLING" not in all_features:
                scores[pattern] = scores.get(pattern, 0) - 0.3

    return scores


def apply_architecture(scores, architecture):
    arch_patterns = ARCH_RULES.get(architecture.upper(), {})

    for pattern, boost in arch_patterns.items():
        if pattern in scores and scores[pattern] > 0.2:
            scores[pattern] += boost * 0.5

    return scores


def normalize_scores(scores):
    if not scores:
        return scores

    values = list(scores.values())

    max_score = max(values)

    # 💣 الحالة دي معناها: مفيش منافسة (واحد بس > 0)
    non_zero = [v for v in values if v > 0]

    if len(non_zero) <= 1:
        return scores  # ❌ ما نعملش normalize

    if max_score == 0:
        return scores

    return {k: round(v / max_score, 3) for k, v in scores.items()}


def get_top_patterns(scores, top_n=3):
    sorted_patterns = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    filtered = [(p, s) for p, s in sorted_patterns if s > 0.05]
    return filtered[:top_n]


def explain_scores(features, scores):
    explanation = {}

    for pattern, categories in RULES.items():
        matched = []

        for f in categories.get("core", []):
            if features.get(f, 0) > 0:
                matched.append(f"[CORE] {f}")

        for f in categories.get("generic", []):
            if features.get(f, 0) > 0:
                matched.append(f"[GENERIC] {f}")

        if pattern in scores:
            explanation[pattern] = matched

    return explanation


def classify_confidence(score):
    if score >= 0.75:
        return "STRONG"
    elif score >= 0.4:
        return "MEDIUM"
    else:
        return "WEAK"