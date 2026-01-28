# ai/methods/weighted_method.py

def compute_total_nfr_weight(freq_norm, must_norm, importance):
    total_weight = {}
    for nfr in freq_norm:
        total_weight[nfr] = (
            0.333 * freq_norm.get(nfr, 0) +
            0.333 * must_norm.get(nfr, 0) +
            0.333 * importance.get(nfr, 0)
        )

    s = sum(total_weight.values()) or 1
    return {k: round(v / s, 4) for k, v in total_weight.items()}
