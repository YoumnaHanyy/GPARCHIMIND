import pandas as pd
from ai.methods.strategy import ArchitectureAnalysisStrategy



def run_ordinal_method(predictions, architecture_data, top_k=5):
    if not predictions:
        raise ValueError("No NFR predictions found for Ordinal Method")

    pred_df = pd.DataFrame(predictions)

    # 🔒 Safety check
    required_pred_cols = {"predicted_type", "predicted_level"}
    if not required_pred_cols.issubset(pred_df.columns):
        raise KeyError(
            f"Prediction data missing required columns: {required_pred_cols}"
        )

    pred_df = pred_df.rename(columns={
        "predicted_type": "Type",
        "predicted_level": "Level"
    })

    arch_df = pd.DataFrame(architecture_data)

    # 🔒 Safety check
    required_arch_cols = {"Type", "Level", "Architecture"}
    if not required_arch_cols.issubset(arch_df.columns):
        raise KeyError(
            f"Architecture dataset missing required columns: {required_arch_cols}"
        )

    matches = pred_df.merge(
        arch_df,
        on=["Type", "Level"],
        how="inner"
    )

    if matches.empty:
        return []

    style_scores = matches["Architecture"].value_counts().head(top_k)

    return [
        {"architecture": arch, "matched_nfrs": int(score)}
        for arch, score in style_scores.items()
    ]


class OrdinalMethod(ArchitectureAnalysisStrategy):
    def run(self, predictions, architecture_data, top_k=5):
        return run_ordinal_method(predictions, architecture_data, top_k)