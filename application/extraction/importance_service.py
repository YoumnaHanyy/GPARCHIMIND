# application/extraction/importance_service.py

import PyPDF2
import nltk
from collections import Counter

nltk.download("punkt", quiet=True)

def compute_importance_from_srs_pdf(pdf_path: str):
    """
    Compute importance score for each NFR type
    based on its frequency in the entire SRS document
    """

    # 1️⃣ Read PDF
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    # 2️⃣ Split into sentences
    sentences = nltk.sent_tokenize(text)
    total_sentences = len(sentences) or 1

    # ⚠️ IMPORTANT
    # هنا إحنا بنفترض إن الـ predicted_type
    # اتخزن قبل كده في non_functional_requirements.json
    from ai.inference.predict_type_level import predict_nfr_type_only

    counts = Counter()

    # 3️⃣ Predict NFR type for each sentence
    for sentence in sentences:
        try:
            nfr_type = predict_nfr_type_only(sentence)
            counts[nfr_type] += 1
        except Exception:
            continue

    # 4️⃣ Normalize
    importance = {
        k: round(v / total_sentences, 4)
        for k, v in counts.items()
    }

    return importance
