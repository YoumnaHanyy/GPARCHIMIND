from collections import Counter


def compute_nfr_statistics(nfr_predictions):
    """
    returns ONLY freq_norm and must_norm
    """


    freq_counts = Counter(
        item["predicted_type"] for item in nfr_predictions
    )

    max_freq = max(freq_counts.values()) or 1
    freq_norm = {
        k: v / max_freq for k, v in freq_counts.items()
    }

    must_scores = {}
    for item in nfr_predictions:
        nfr = item["predicted_type"]
        desc = item["predicted_level"].lower()

        if "high" in desc:
            weight = 3
        elif "medium" in desc:
            weight = 2
        elif "low" in desc:
            weight = 1
        else:
            weight = 0.5

        must_scores[nfr] = must_scores.get(nfr, 0) + weight

    max_must = max(must_scores.values()) or 1
    must_norm = {
        k: v / max_must for k, v in must_scores.items()
    }

    # ---------- Importance ----------
    total = sum(freq_counts.values()) or 1
    importance = {
         k: v / total for k, v in freq_counts.items()
    }

    return freq_norm, must_norm, importance

