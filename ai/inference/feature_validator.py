from ai.utils.feature_keywords import FEATURE_KEYWORDS
from ai.utils.text_utils import normalize_text


def match_keyword(word, text):
    return word in text or word.replace("-", " ") in text


def validate_features(features, text):
    text = normalize_text(text)

    strict_features = {
        "EVENT_DRIVEN": ["event driven", "publish", "subscribe", "message queue"],
        "REAL_TIME": ["real time", "live"],
        "FLEXIBLE_CREATION": ["factory", "builder"],
        "DYNAMIC_BEHAVIOR": ["strategy", "runtime"]
    }

    for feature, keywords in FEATURE_KEYWORDS.items():

        if feature not in features:
            continue

        # 🧠 count evidence strength
        matches = [word for word in keywords if word in text]
        count = len(matches)

        # 🔥 dynamic scoring
        if count == 0:
            features[feature] *= 0.8

        elif count == 1:
            features[feature] = max(features[feature], 0.5)

        elif count == 2:
            features[feature] = max(features[feature], 0.7)

        elif count >= 3:
            features[feature] = max(features[feature], 0.85)

        # 🚫 منع 1.0 إلا بشروط قوية
        if features[feature] > 0.95:
            if count < 4:   # لازم 4 keywords على الأقل
                features[feature] = 0.9

    return features