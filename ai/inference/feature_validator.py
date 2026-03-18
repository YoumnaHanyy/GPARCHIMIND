from ai.utils.feature_keywords import FEATURE_KEYWORDS

def validate_features(features, text):
    text = text.lower()

    # 🔴 strict features (لازم دليل صريح)
    strict_features = {
    "EVENT_DRIVEN": ["event-driven", "publish", "subscribe", "message queue"],  # ❌ شيلنا notification
    "REAL_TIME": ["real-time", "live", "instant"],
    "FLEXIBLE_CREATION": ["factory", "builder"],
    "DYNAMIC_BEHAVIOR": ["strategy", "switch"]
}

    # 🔥 strict validation لكل feature لوحده
    for feature, keywords in strict_features.items():
        if not any(word in text for word in keywords):
            features[feature] = 0.3

    return features