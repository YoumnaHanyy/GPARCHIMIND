import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from infrastructure.database import db
from collections import defaultdict

MODEL_DIR = "hanawahab/trained_nfr_binary_model"

NFR_ORDER = ["PE", "SC", "MN", "A", "SE", "US", "PO", "O"]

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


def predict_binary_architecture():
    # =============================
    # 1️⃣ Load predicted NFRs
    # =============================
    nfr_collection = db["nfr_predictions"]

    extracted_nfrs = list(
        nfr_collection.find(
            {}, {"_id": 0, "description": 1, "predicted_type": 1}
        )
    )

    def predict(sentence):
        tokens = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )
        with torch.no_grad():
            output = model(**tokens)
        return int(torch.argmax(output.logits).item()) 

    # =============================
    # 2️⃣ Build SRS binary vector
    # =============================
    srs_vector = {k: 0 for k in NFR_ORDER}

    for item in extracted_nfrs:
        nfr_type = item.get("predicted_type")
        if nfr_type in NFR_ORDER:
            srs_vector[nfr_type] = predict(item["description"])

    srs_vec = np.array([int(srs_vector[k]) for k in NFR_ORDER])

    # =============================
    # 3️⃣ Load architectures from Mongo
    # =============================
    arch_collection = db["ArchitectureDataset"]

    raw_archs = list(
        arch_collection.find(
            {},
            {"_id": 0, "Architecture": 1, "Type": 1, "label": 1}
        )
    )

    arch_vectors = defaultdict(lambda: {k: 0 for k in NFR_ORDER})

    for row in raw_archs:
        arch_name = row.get("Architecture")
        nfr_type = row.get("Type")
        label = row.get("label")

        # 👇 critical fix
        if (
            arch_name
            and nfr_type in NFR_ORDER
            and label is not None
        ):
            arch_vectors[arch_name][nfr_type] = int(label)

    # =============================
    # 4️⃣ Score architectures
    # =============================
    results = []

    for arch_name, vec in arch_vectors.items():
        arch_vec = np.array([int(vec[k]) for k in NFR_ORDER])

        diff = np.sum(np.abs(arch_vec - srs_vec))
        score = round(1 - (diff / len(NFR_ORDER)), 3)

        results.append({
            "architecture": arch_name,
            "score": score
        })

    # =============================
    # 5️⃣ Sort & return
    # =============================
    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "srs_vector": srs_vector,
        "top_architectures": results[:5]
    }