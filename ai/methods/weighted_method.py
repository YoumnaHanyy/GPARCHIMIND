# ai/methods/weighted_method.py
from ai.methods.strategy import ArchitectureAnalysisStrategy



def compute_total_nfr_weight(freq_norm, must_norm, importance):
    total_weight = {}
    for nfr in freq_norm:
        total_weight[nfr] = (
            0.333 * freq_norm.get(nfr, 0) +
            0.333 * must_norm.get(nfr, 0) +
            0.333 * importance.get(nfr, 0)
        )
    
    s = sum(total_weight.values()) or 1
    print("TOTAL WEIGHTS:", total_weight)
    return {k: round(v / s, 4) for k, v in total_weight.items()}



class WeightedMethod(ArchitectureAnalysisStrategy):
    def run(self, freq_norm, must_norm, importance):
        return compute_total_nfr_weight(freq_norm, must_norm, importance)
