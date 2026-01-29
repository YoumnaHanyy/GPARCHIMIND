import numpy as np
import torch
import pandas as pd

from transformers import BertTokenizer, BertForSequenceClassification

# MongoDB
from infrastructure.database import db


# ============================================================
# 1️⃣ Load Binary BERT Model
# ============================================================

MODEL_DIR = "trained_nfr_binary_model"

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


# ============================================================
# 2️⃣ Binary Prediction for One Sentence
# ============================================================

def predict_binary(sentence: str) -> int:
    """
    Predict binary label for one sentence
    return: 0 or 1
    """
    tokens = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    with torch.no_grad():
        output = model(**tokens)

    return torch.argmax(output.logits).item()


# ============================================================
# 3️⃣ Load NFR Sentences from MongoDB
# ============================================================

def load_nfr_sentences():
    """
    Reads NFR requirements from MongoDB
    Collection: merged_NFR_cleaned_no_dots
    """
    collection = db["merged_NFR_cleaned_no_dots"]
    docs = list(collection.find({}, {"_id": 0, "Requirement": 1}))
    return [d["Requirement"] for d in docs]


# ============================================================
# 4️⃣ Build Binary Vector
# ============================================================

NFR_ORDER = ["PE", "SC", "MN", "A", "SE", "US", "PO", "O"]

def build_binary_vector(sentences):
    """
    Convert SRS sentences into binary NFR vector
    """
    vector = {k: 0 for k in NFR_ORDER}

    for s in sentences:
        for nfr in NFR_ORDER:
            if nfr.lower() in s.lower():
                vector[nfr] = predict_binary(s)

    return vector


# ============================================================
# 5️⃣ Load Architecture Dataset from MongoDB
# ============================================================

def load_architecture_dataset():
    """
    Reads architecture dataset from MongoDB
    Collection: ArchitectureDataset
    """
    collection = db["ArchitectureDataset"]
    docs = list(collection.find({}, {"_id": 0}))
    return pd.DataFrame(docs)


# ============================================================
# 6️⃣ Compute Architecture Scores
# ============================================================

def compute_architecture_scores(binary_vector, arch_df):
    """
    Compare SRS vector with architecture vectors
    """
    results = []

    for _, row in arch_df.iterrows():
        arch_name = row["Architecture Style"]

        arch_vec = row[NFR_ORDER].values.astype(int)
        srs_vec = np.array([binary_vector[k] for k in NFR_ORDER])

        diff = np.sum(np.abs(arch_vec - srs_vec))
        score = 1 - (diff / len(NFR_ORDER))

        results.append({
            "architecture": arch_name,
            "score": round(float(score), 4)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ============================================================
# 7️⃣ Main Binary Method Pipeline
# ============================================================

def run_binary_method():
    """
    Full Binary Method pipeline
    """
    sentences = load_nfr_sentences()
    binary_vector = build_binary_vector(sentences)

    arch_df = load_architecture_dataset()
    scores = compute_architecture_scores(binary_vector, arch_df)

    return {
        "binary_vector": binary_vector,
        "top_5_architectures": scores[:5],
        "best_architecture": scores[0] if scores else None
    }


# ============================================================
# 8️⃣ Local Testing
# ============================================================

if __name__ == "__main__":
    result = run_binary_method()

    print("\n=== Binary Vector ===")
    print(result["binary_vector"])

    print("\n=== Top 5 Architectures ===")
    for arch in result["top_5_architectures"]:
        print(arch)

    print("\n=== Best Architecture ===")
    print(result["best_architecture"])
