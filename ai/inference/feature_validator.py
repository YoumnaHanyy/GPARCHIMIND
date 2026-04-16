from ai.utils.feature_keywords import FEATURE_KEYWORDS
from ai.utils.text_utils import normalize_text


def match_keyword(word, text):
    return (
        word in text or
        word.replace("-", " ") in text or
        word.replace(" ", "") in text or
        any(w in text for w in word.split())
    )


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

        # 🧠 keyword matching
        matches = [word for word in keywords if match_keyword(word, text)]
        count = len(matches)

        current = features[feature]

        # 🔥 boost تدريجي (مش قفزات)
        if count == 1:
            current = max(current, 0.5)

        elif count == 2:
            current = max(current, 0.6)

        elif count >= 3:
            current = min(0.85, current + 0.15)

        # 🧠 weak signal correction (بس لو AI ضعيف جدًا)
        if count == 0:
            current = min(current, 0.2)

        # 🔒 strict features control
        if feature in strict_features:
            strict_words = strict_features[feature]
            if not any(match_keyword(w, text) for w in strict_words):
                current = min(current, 0.5)

        # 🚫 منع القيم العالية بدون evidence كفاية
        if current > 0.9 and count < 3:
            current = 0.85

        # 🎯 final cap
        current = min(current, 0.9)

        features[feature] = round(current, 2)

    return features


