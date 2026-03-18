import json
import re
from ai.utils.feature_keywords import FEATURE_KEYWORDS


def parse_response(response):
    # 🧹 تنظيف response
    response = re.sub(r",\s*,+", ",", response)

    try:
        return json.loads(response)
    except:
        match = re.search(r"\{.*\}", response, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass

        # 🔥 fallback ذكي
        text = response.lower().replace("-", " ")

        features = {key: 0.0 for key in FEATURE_KEYWORDS.keys()}

        for feature, keywords in FEATURE_KEYWORDS.items():
            for word in keywords:
                if word in text or word.replace(" ", "") in text:
                    features[feature] = max(features[feature], 0.8)

        return features