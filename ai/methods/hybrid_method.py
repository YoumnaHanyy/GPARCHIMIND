import collections

# ============================================================
# 1️⃣ Hybrid Configuration
# ============================================================

DEFAULT_WEIGHTS = {
    "ordinal": 0.3,
    "binary": 0.3,
    "weighted": 0.4
}


# ============================================================
# 2️⃣ Normalize Scores Helper
# ============================================================

def normalize_scores(score_dict):
    """
    Normalize scores to range [0,1]
    """
    if not score_dict:
        return {}

    max_val = max(score_dict.values()) or 1
    return {k: round(v / max_val, 4) for k, v in score_dict.items()}


# ============================================================
# 3️⃣ Convert Method Outputs to Comparable Scores
# ============================================================

def scores_from_ordinal(ordinal_result):
    """
    ordinal_result: list of architectures (ranked)
    Higher rank = higher score
    """
    scores = {}
    total = len(ordinal_result)

    for i, arch in enumerate(ordinal_result):
        scores[arch] = total - i

    return normalize_scores(scores)


def scores_from_binary(binary_result):
    """
    binary_result: output from binary_method.run_binary_method
    """
    scores = {}
    for item in binary_result["top_5_architectures"]:
        scores[item["architecture"]] = item["score"]

    return normalize_scores(scores)


def scores_from_weighted(weighted_result):
    """
    weighted_result: output from weighted method
    """
    scores = {}
    for item in weighted_result["top_architectures"]:
        scores[item["Architecture"]] = item["Score"]

    return normalize_scores(scores)


# ============================================================
# 4️⃣ Hybrid Score Computation
# ============================================================

def compute_hybrid_scores(
    ordinal_result,
    binary_result,
    weighted_result,
    weights=DEFAULT_WEIGHTS
):
    """
    Combine all methods into one hybrid ranking
    """
    final_scores = collections.defaultdict(float)

    ordinal_scores = scores_from_ordinal(ordinal_result)
    binary_scores = scores_from_binary(binary_result)
    weighted_scores = scores_from_weighted(weighted_result)

    all_architectures = set(
        list(ordinal_scores.keys()) +
        list(binary_scores.keys()) +
        list(weighted_scores.keys())
    )

    for arch in all_architectures:
        final_scores[arch] += weights["ordinal"] * ordinal_scores.get(arch, 0)
        final_scores[arch] += weights["binary"] * binary_scores.get(arch, 0)
        final_scores[arch] += weights["weighted"] * weighted_scores.get(arch, 0)

    # Normalize final scores
    max_score = max(final_scores.values()) or 1
    final_scores = {
        k: round(v / max_score, 4) for k, v in final_scores.items()
    }

    return sorted(
        [{"architecture": k, "score": v} for k, v in final_scores.items()],
        key=lambda x: x["score"],
        reverse=True
    )


# ============================================================
# 5️⃣ Main Hybrid Method
# ============================================================

def run_hybrid_method(
    ordinal_result,
    binary_result,
    weighted_result,
    top_k=5
):
    """
    Final Hybrid Architecture Recommendation
    """
    hybrid_scores = compute_hybrid_scores(
        ordinal_result,
        binary_result,
        weighted_result
    )

    return {
        "top_5_architectures": hybrid_scores[:top_k],
        "best_architecture": hybrid_scores[0] if hybrid_scores else None
    }


# ============================================================
# 6️⃣ Local Testing
# ============================================================

if __name__ == "__main__":

    # Dummy test data
    ordinal = ["Layered", "Microservices", "Client-Server"]
    binary = {
        "top_5_architectures": [
            {"architecture": "Layered", "score": 0.8},
            {"architecture": "Microservices", "score": 0.6}
        ]
    }
    weighted = {
        "top_architectures": [
            {"Architecture": "Microservices", "Score": 0.9},
            {"Architecture": "Layered", "Score": 0.85}
        ]
    }

    result = run_hybrid_method(ordinal, binary, weighted)

    print("\n=== Hybrid Top Architectures ===")
    for r in result["top_5_architectures"]:
        print(r)

    print("\n=== Best Architecture ===")
    print(result["best_architecture"])
