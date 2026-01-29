"""
Text utilities:
- Cleaning
- Normalization
- Simple keyword helpers
"""

import re
import json, re, logging
from typing import List, Dict

logger = logging.getLogger("archimind")

# ============================================================
# 1️⃣ Basic text cleaning
# ============================================================

def clean_text(text: str) -> str:
    """
    Basic text normalization:
    - lowercasing
    - remove extra spaces
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# 2️⃣ Remove special characters
# ============================================================

def remove_special_chars(text: str) -> str:
    """
    Remove non-alphanumeric characters except punctuation
    """
    if not text:
        return ""

    return re.sub(r"[^a-zA-Z0-9\s.,;:!?]", "", text)


# ============================================================
# 3️⃣ Requirement strength detection
# ============================================================

def detect_requirement_strength(text: str) -> str:
    """
    Detect requirement strength keywords
    """
    t = text.lower()

    if "must" in t:
        return "MUST"
    if "shall" in t:
        return "SHALL"
    if "should" in t:
        return "SHOULD"
    return "WEAK"


# ============================================================
# 4️⃣ Combined preprocess
# ============================================================

def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline
    """
    text = clean_text(text)
    text = remove_special_chars(text)
    return text

def parse_model_json(output_text: str) -> List[Dict]:
    if not output_text:
        return []

    cleaned = re.sub(r"```json\s*", "", output_text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)

    try:
        result = json.loads(cleaned.strip())
        if isinstance(result, list):
            return result
    except:
        pass

    match = re.search(r'\[[\s\S]*\]', cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    logger.warning("Failed to parse JSON array")
    return []


def extract_json_from_model_output(output: str) -> str:
    cleaned = re.sub(r"```json\s*", "", output)
    cleaned = re.sub(r"```", "", cleaned)
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    stack = []
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            stack.append("{")
        elif cleaned[i] == "}":
            stack.pop()
            if not stack:
                return cleaned[start:i + 1]

    raise ValueError("Unbalanced JSON")