from ai.methods.strategy import ArchitectureAnalysisStrategy



def extract_behavior_signals(functional_reqs):
    signals = {
        "user_driven": 0,
        "event_triggered": 0,
        "workflow_oriented": 0,
        "extensible": 0,
        "integration_heavy": 0,
        "data_centric": 0
    }

    for fr in functional_reqs:
        text = (fr.get("title","") + " " + fr.get("description","")).lower()

        if any(k in text for k in ["user", "actor", "login", "role"]):
            signals["user_driven"] += 1

        if any(k in text for k in ["event", "notify", "trigger", "when"]):
            signals["event_triggered"] += 1

        if any(k in text for k in ["step", "process", "workflow"]):
            signals["workflow_oriented"] += 1

        if any(k in text for k in ["plugin", "extend", "add new"]):
            signals["extensible"] += 1

        if any(k in text for k in ["external", "third-party", "api"]):
            signals["integration_heavy"] += 1

        if any(k in text for k in ["store", "update", "database"]):
            signals["data_centric"] += 1

    return signals

ARCH_PROFILES = {
    "mvc": ["user_driven"],
    "client_server": ["user_driven", "integration_heavy"],
    "event_driven": ["event_triggered"],
    "pipe_filter": ["workflow_oriented"],
    "microkernel": ["extensible"],
    "soa": ["integration_heavy"],
    "repository": ["data_centric"],
    "layered": ["user_driven", "workflow_oriented"]
}

def choose_architecture(functional_reqs):
    signals = extract_behavior_signals(functional_reqs)

    scores = {}
    reasons = {}

    for arch, needed_signals in ARCH_PROFILES.items():
        score = 0
        arch_reasons = []

        for s in needed_signals:
            score += signals[s]
            if signals[s] > 0:
                arch_reasons.append(
                    f"The system shows {s.replace('_',' ')} behavior in functional requirements."
                )

        if score > 0:
            scores[arch] = score
            reasons[arch] = arch_reasons

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    max_score = ranked[0][1] if ranked else 1

    return {
        "top_architectures": [
            {
                "Architecture": arch,
                "Score": round(score / max_score, 4),
                "Reason": " ".join(reasons[arch])
            }
            for arch, score in ranked
        ],
        "signals": signals
    }

class FunctionalMethod(ArchitectureAnalysisStrategy):
    def run(self, functional_requirements):
        return choose_architecture(functional_requirements)