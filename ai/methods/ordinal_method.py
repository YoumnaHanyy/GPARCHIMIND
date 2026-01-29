import pandas as pd
import json
from collections import Counter


# ============================================================
# 1️⃣ Load Predictions (Type + Level)
# ============================================================

def load_predictions(prediction_file: str):
    """
    Load predicted NFR Type + Level from JSON
    """
    with open(prediction_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return pd.DataFrame(data)


# ============================================================
# 2️⃣ Load Architecture Dataset
# ============================================================

def load_architecture_dataset(csv_path: str):
    """
    Load architecture mapping dataset
    """
    df = pd.read_csv(csv_path)

    # Normalize column names
    df = df.rename(columns={
        "architecture_style": "Architecture",
        "architecture style": "Architecture",
        "architecture": "Architecture"
    })

    return df


# ============================================================
# 3️⃣ Ordinal Scoring Logic
# ============================================================

def compute_ordinal_scores(pred_df: pd.DataFrame, arch_df: pd.DataFrame):
    """
    Count matches of (Type, Level) for each architecture
    """
    merged = pred_df.merge(
        arch_df,
        on=["Type", "Level"],
        how="inner"
    )

    scores = Counter(merged["Architecture"])
    return scores


# ============================================================
# 4️⃣ Run Ordinal Method
# ============================================================

def run_ordinal_method(
    prediction_file="nfr_predictions_type_level.json",
    architecture_dataset="ArchitectureDataset.csv",
    top_k=5
):
    """
    Main Ordinal Method Runner
    """
    pred_df = load_predictions(prediction_file)

    # Rename for safety
    pred_df = pred_df.rename(columns={
        "predicted_type": "Type",
        "predicted_level": "Level"
    })

    arch_df = load_architecture_dataset(architecture_dataset)

    scores = compute_ordinal_scores(pred_df, arch_df)

    ranked_architectures = [
        arch for arch, _ in scores.most_common(top_k)
    ]

    return ranked_architectures


# ============================================================
# 5️⃣ Local Testing
# ============================================================

if __name__ == "__main__":
    result = run_ordinal_method(
        prediction_file="nfr_predictions_type_level.json",
        architecture_dataset="ArchitectureDataset.csv"
    )

    print("\n=== Ordinal Method – Top Architectures ===")
    for i, arch in enumerate(result, 1):
        print(f"{i}. {arch}")
