import json
import pandas as pd
import numpy as np
from collections import defaultdict


# ============================================================
# 1️⃣ Requirement Strength (MUST / SHALL / SHOULD)
# ============================================================

def requirement_strength(description: str) -> float:
    s = description.lower()
    if "must" in s:
        return 3.0
    if "shall" in s:
        return 2.0
    if "should" in s:
        return 1.0
    return 0.5


# ============================================================
# 2️⃣ Compute Frequency + Must Score
# ============================================================

def compute_frequency_and_must(extracted_nfrs, predict_nfr_fn):
    freq = defaultdict(int)
    must_score = defaultdict(float)

    for item in extracted_nfrs:
        desc = item["description"]
        nfr_type = predict_nfr_fn(desc)

        freq[nfr_type] += 1
        must_score[nfr_type] += requirement_strength(desc)

    # Normalize
    max_freq = max(freq.values()) if freq else 1
    max_must = max(must_score.values()) if must_score else 1

    freq_norm = {k: v / max_freq for k, v in freq.items()}
    must_norm = {k: v / max_must for k, v in must_score.items()}

    return freq_norm, must_norm


# ============================================================
# 3️⃣ Compute Importance Score (from full SRS)
# ============================================================

def compute_importance_score(predicted_nfrs):
    total = len(predicted_nfrs) or 1
    counts = defaultdict(int)

    for nfr in predicted_nfrs:
        counts[nfr] += 1

    importance = {k: v / total for k, v in counts.items()}
    return importance


# ============================================================
# 4️⃣ Total NFR Weight
# ============================================================

def compute_total_weights(freq_norm, must_norm, importance,
                          weights=(0.333, 0.333, 0.333)):

    total = {}
    for nfr in freq_norm.keys():
        total[nfr] = (
            weights[0] * freq_norm.get(nfr, 0) +
            weights[1] * must_norm.get(nfr, 0) +
            weights[2] * importance.get(nfr, 0)
        )

    s = sum(total.values()) or 1
    return {k: round(v / s, 4) for k, v in total.items()}


# ============================================================
# 5️⃣ Compute Architecture Scores
# ============================================================

def compute_architecture_scores(total_weights, architecture_csv):
    df = pd.read_csv(architecture_csv)
    scores = defaultdict(float)

    for _, row in df.iterrows():
        arch = row["Architecture"]
        nfr = row["Type"]
        level_norm = row.get("LevelNorm", 1)

        if nfr in total_weights:
            scores[arch] += level_norm * total_weights[nfr]

    s = sum(scores.values()) or 1
    return {k: round(v / s, 4) for k, v in scores.items()}


# ============================================================
# 6️⃣ Run Weighted Method
# ============================================================

def run_weighted_method(
    extracted_nfr_file="non_functional_requirements.json",
    architecture_csv="ArchitectureDataset2.csv",
    predict_nfr_fn=None,
    top_k=5
):
    """
    predict_nfr_fn: function that takes description -> predicted NFR type
    """

    if predict_nfr_fn is None:
        raise ValueError("predict_nfr_fn must be provided")

    with open(extracted_nfr_file, "r", encoding="utf-8") as f:
        extracted_nfrs = json.load(f)

    # Predict NFRs
    predicted_nfrs = [
        predict_nfr_fn(item["description"]) for item in extracted_nfrs
    ]

    freq_norm, must_norm = compute_frequency_and_must(
        extracted_nfrs, predict_nfr_fn
    )

    importance = compute_importance_score(predicted_nfrs)

    total_weights = compute_total_weights(
        freq_norm, must_norm, importance
    )

    arch_scores = compute_architecture_scores(
        total_weights, architecture_csv
    )

    ranked = sorted(
        arch_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return {
        "normalized_frequency": freq_norm,
        "normalized_must_score": must_norm,
        "importance_score": importance,
        "total_nfr_weights": total_weights,
        "top_architectures": [
            {"Architecture": k, "Score": v} for k, v in ranked
        ]
    }


# ============================================================
# 7️⃣ Local Test
# ============================================================

if __name__ == "__main__":
    print("Weighted Method module loaded successfully")
