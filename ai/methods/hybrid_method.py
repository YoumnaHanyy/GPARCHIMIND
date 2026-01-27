# ai/methods/hybrid_method.py

def normalize_scores(score_dict):
    """
    Normalize a dict of scores to range [0,1]
    """
    if not score_dict:
        return {}

    max_val = max(score_dict.values()) or 1
    return {k: v / max_val for k, v in score_dict.items()}


def hybrid_aggregation(functional, ordinal, binary, weighted):
    """
    Combine Functional, Ordinal, Binary, and Weighted methods
    into one final ranked architecture list.
    """

    final_scores = {}
    architectures = set()

    # =====================================================
    # 1️⃣ Collect architectures safely
    # =====================================================

    # ---- Ordinal ----
    if isinstance(ordinal, list):
        for item in ordinal:
            if isinstance(item, dict) and "Architecture" in item:
                architectures.add(item["Architecture"])

    # ---- Binary ----
    if isinstance(binary, list):
        for item in binary:
            if isinstance(item, dict) and "architecture" in item:
                architectures.add(item["architecture"])

    # ---- Weighted ----
    if isinstance(weighted, dict):
        for item in weighted.get("top_architectures", []):
            if isinstance(item, dict) and "Architecture" in item:
                architectures.add(item["Architecture"])

    # ---- Functional ----
    functional_scores = {}
    if isinstance(functional, dict):
        for a in functional.get("top_architectures", []):
            if isinstance(a, dict):
                functional_scores[a["Architecture"]] = a.get("Score", 0)

    # =====================================================
    # 2️⃣ Max values for normalization
    # =====================================================

    max_f = max(functional_scores.values()) if functional_scores else 1

    ordinal_values = [
        a.get("MatchedNFRs", 0)
        for a in ordinal
        if isinstance(a, dict)
    ] if isinstance(ordinal, list) else []
    max_o = max(ordinal_values) if ordinal_values else 1

    binary_values = [
        a.get("score", 0)
        for a in binary
        if isinstance(a, dict)
    ] if isinstance(binary, list) else []
    max_b = max(binary_values) if binary_values else 1

    weighted_values = [
        a.get("Score", 0)
        for a in weighted.get("top_architectures", [])
        if isinstance(a, dict)
    ] if isinstance(weighted, dict) else []
    max_w = max(weighted_values) if weighted_values else 1

    # =====================================================
    # 3️⃣ Compute final score per architecture
    # =====================================================

    for arch in architectures:

        # ---------- Functional ----------
        raw_f = functional_scores.get(arch, 0)
        s_f = raw_f / max_f if max_f else 0

        # ---------- Ordinal (count → score) ----------
        raw_o = 0
        if isinstance(ordinal, list):
            for a in ordinal:
                if isinstance(a, dict) and a.get("Architecture") == arch:
                    raw_o = a.get("MatchedNFRs", 0)
                    break
        s_o = raw_o / max_o if max_o else 0

        # ---------- Binary ----------
        raw_b = 0
        if isinstance(binary, list):
            for a in binary:
                if isinstance(a, dict) and a.get("architecture") == arch:
                    raw_b = a.get("score", 0)
                    break
        s_b = raw_b 

        # ---------- Weighted ----------
        raw_w = 0
        if isinstance(weighted, dict):
            for a in weighted.get("top_architectures", []):
                if isinstance(a, dict) and a.get("Architecture") == arch:
                    raw_w = a.get("Score", 0)
                    break
        s_w = raw_w / max_w if max_w else 0

        # ---------- Final Weighted Sum ----------
        final_scores[arch] = (
            0.20 * s_f +
            0.25 * s_o +
            0.20 * s_b +
            0.35 * s_w
        )

    # =====================================================
    # 4️⃣ Normalize final scores & return top 5
    # =====================================================

    final_scores = normalize_scores(final_scores)

    return sorted(
        [
            {
                "Architecture": arch,
                "FinalScore": round(score * 100, 2)
            }
            for arch, score in final_scores.items()
        ],
        key=lambda x: x["FinalScore"],
        reverse=True
    )[:5]
