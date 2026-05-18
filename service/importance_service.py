import PyPDF2
import nltk
import re
from collections import Counter
from ai.inference.predict_type_level import predict_nfr_type_only

nltk.download("punkt", quiet=True)

# 🔑 Keywords لكل NFR
NFR_KEYWORDS = {
    "Performance": ["performance", "speed", "latency", "response time"],
    "Security": ["security", "secure", "encryption", "authentication"],
    "Scalability": ["scalability", "scale", "load", "concurrent"],
    "Usability": ["usability", "easy", "user-friendly", "interface"]
}

def compute_importance_from_srs_pdf(pdf_path: str):
    text = ""

    # 1️⃣ Read PDF
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    # =========================
    # 🔵 PART 1: MODEL BASED
    # =========================
    sentences = nltk.sent_tokenize(text)
    total_sentences = len(sentences) or 1

    # ⚠️ IMPORTANT
    # هنا إحنا بنفترض إن الـ predicted_type
    # اتخزن قبل كده في non_functional_requirements.json
    from ai.inference.predict_type_level import predict_nfr_type_only

    model_counts = Counter()

    for sentence in sentences:
        try:
            nfr_type = predict_nfr_type_only(sentence)
            model_counts[nfr_type] += 1
        except Exception:
            continue

    model_scores = {
        k: v / total_sentences
        for k, v in model_counts.items()
    }

    # =========================
    # 🟡 PART 2: KEYWORD BASED
    # =========================
    text_lower = text.lower()
    keyword_counts = Counter()

    for nfr_type, keywords in NFR_KEYWORDS.items():
        for kw in keywords:
            matches = len(re.findall(rf"\b{kw}\b", text_lower))
            keyword_counts[nfr_type] += matches

    total_keywords = sum(keyword_counts.values()) or 1

    keyword_scores = {
        k: v / total_keywords
        for k, v in keyword_counts.items()
    }

    # =========================
    # 🔴 PART 3: HYBRID MERGE
    # =========================
    final_scores = {}
    all_keys = set(model_scores) | set(keyword_scores)

    for k in all_keys:
        m = model_scores.get(k, 0)
        kw = keyword_scores.get(k, 0)

        final_scores[k] = round((0.7 * m) + (0.3 * kw), 4)

    return final_scores