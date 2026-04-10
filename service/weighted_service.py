# application/extraction/weighted_service.py

from ai.methods.weighted_method import compute_total_nfr_weight
from infrastructure.repositories.architecture_repository import get_architecture_dataset


def execute_weighted_method(freq_norm, must_norm, importance):
    total_weights = compute_total_nfr_weight(freq_norm, must_norm, importance)

    arch_data = get_architecture_dataset()  # 👈 list of dicts
    scores = {}

    for row in arch_data:
        arch = row["Architecture"]
        nfr = row["Type"]
        level_norm_raw = row.get("LevelNorm", 0)

        if level_norm_raw is None:
         level_norm = 0.0
        elif hasattr(level_norm_raw, "to_decimal"):
         level_norm = float(level_norm_raw.to_decimal())
        else:
         level_norm = float(level_norm_raw)
  # safety

        if nfr in total_weights:
            scores[arch] = scores.get(arch, 0) + (
                level_norm * total_weights[nfr]
            )
        print("ARCH:", arch, "NFR:", nfr, "LEVEL:", level_norm)
    return {
    "top_architectures": sorted(
        [{"Architecture": k, "Score": round(v, 4)} for k, v in scores.items()],
        key=lambda x: x["Score"],
        reverse=True
    )[:5]
}
#comment
