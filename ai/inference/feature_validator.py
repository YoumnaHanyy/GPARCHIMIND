from ai.utils.feature_keywords import FEATURE_KEYWORDS
def validate_features(features, text):
    text = text.lower()

    # 🔴 features لازم دليل صريح
    strict_features = {
        "EVENT_DRIVEN": ["event", "trigger", "publish", "subscribe","notification","notifications"],
        "REAL_TIME": ["real-time", "live", "instant"],
        "FLEXIBLE_CREATION": ["create different", "dynamic creation"],
        "DYNAMIC_BEHAVIOR": ["strategy", "switch behavior"]
    }

    for feature, keywords in strict_features.items():
        if not any(word in text for word in keywords):
            features[feature] = 0.0  # 💣 نمنع الهلوسة تمامًا

    return features