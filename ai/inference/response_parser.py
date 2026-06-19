import json
import re
from ai.utils.feature_keywords import FEATURE_KEYWORDS

def _coerce_score(value):
    """Coerce an LLM-emitted feature value to a float in [0.0, 1.0].

    The model sometimes returns nested objects per feature (e.g.
    {"score": 0.9, "reason": "..."}) or strings ("0.85"); upstream code
    assumes a bare number and does sum()/min() on it, which crashes on a dict.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return 0.0
        return 0.0
    if isinstance(value, dict):
        for key in ("score", "confidence", "value", "weight"):
            if key in value:
                return _coerce_score(value[key])
        for v in value.values():
            coerced = _coerce_score(v)
            if coerced:
                return coerced
        return 0.0
    if isinstance(value, list):
        for v in value:
            coerced = _coerce_score(v)
            if coerced:
                return coerced
        return 0.0
    return 0.0


def _sanitize_features(features):
    if not isinstance(features, dict):
        return {key: 0.0 for key in FEATURE_KEYWORDS.keys()}
    return {str(k): _coerce_score(v) for k, v in features.items()}

def parse_response(response):
    # 🧹 تنظيف response
    response = re.sub(r",\s*,+", ",", response)

    try:
        return _sanitize_features(json.loads(response))
    except Exception:
        match = re.search(r"\{.*\}", response, re.DOTALL)

        if match:
            try:
                return _sanitize_features(json.loads(match.group(0)))
            except Exception:
                pass

        # 🔥 fallback ذكي
        text = response.lower().replace("-", " ")

        features = {key: 0.0 for key in FEATURE_KEYWORDS.keys()}

        for feature, keywords in FEATURE_KEYWORDS.items():
            for word in keywords:
                if word in text or word.replace(" ", "") in text:
                   features[feature] = max(features[feature], 0.2)

        return features