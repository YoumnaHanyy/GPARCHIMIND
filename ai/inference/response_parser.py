import json
import re
from ai.utils.feature_keywords import FEATURE_KEYWORDS
FEATURE_KEYS = [
    "EVENT_DRIVEN",
    "REAL_TIME",
    "HIGH_SCALABILITY",
    "LOW_LATENCY",
    "HIGH_EXTENSIBILITY",
    "HIGH_MAINTAINABILITY",
    "FLEXIBLE_CREATION",
    "DYNAMIC_BEHAVIOR",
    "DISTRIBUTED_SYSTEM",
    "FAULT_TOLERANCE",
    "HIGH_SECURITY",
    "MODULARITY",
    "HIGH_COUPLING_RISK"
]


def parse_response(response):
    import json
    import re

    try:
        return json.loads(response)
    except:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        text = response.lower()

        features = {key: 0.0 for key in FEATURE_KEYWORDS.keys()}

        # 🔥 smart keyword matching
        for feature, keywords in FEATURE_KEYWORDS.items():
            for word in keywords:
                if word in text:
                    features[feature] = max(features[feature], 0.8)

        return features