import torch
import pandas as pd
import numpy as np
import json

from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# MongoDB
from infrastructure.database import db


# ============================================================
# 1️⃣ Load Training Data from MongoDB (for Label Encoders)
# ============================================================

def load_training_dataset():
    """
    Collection: merged_NFR_cleaned_no_dots
    Fields: Type, Level
    """
    collection = db["merged_NFR_cleaned_no_dots"]
    docs = list(collection.find({}, {"_id": 0, "Type": 1, "Level": 1}))
    return pd.DataFrame(docs)


df_train = load_training_dataset()

le_type = LabelEncoder()
le_type.fit(df_train["Type"])

le_level = LabelEncoder()
le_level.fit(df_train["Level"])


# ============================================================
# 2️⃣ Load Models + Tokenizer
# ============================================================

TOKENIZER_NAME = "bert-base-uncased"

MODEL_TYPE_DIR = "trained_nfr_type_model"
MODEL_LEVEL_DIR = "trained_nfr_level_model"

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

model_type = BertForSequenceClassification.from_pretrained(MODEL_TYPE_DIR)
model_level = BertForSequenceClassification.from_pretrained(MODEL_LEVEL_DIR)

model_type.eval()
model_level.eval()


# ============================================================
# 3️⃣ Load Extracted NFRs from MongoDB
# ============================================================

def load_extracted_nfrs():
    """
    Collection: extracted_nfrs
    Field: description
    """
    collection = db["extracted_nfrs"]
    docs = list(collection.find({}, {"_id": 0, "title": 1, "description": 1}))
    return docs


# ============================================================
# 4️⃣ Predict Type + Level
# ============================================================

def predict_type_and_level(nfr_list):
    texts = [item["description"] for item in nfr_list]

    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        out_type = model_type(**tokens)
        out_level = model_level(**tokens)

    pred_type_idx = torch.argmax(out_type.logits, dim=1).numpy()
    pred_level_idx = torch.argmax(out_level.logits, dim=1).numpy()

    pred_types = le_type.inverse_transform(pred_type_idx)
    pred_levels = le_level.inverse_transform(pred_level_idx)

    results = []
    for i, item in enumerate(nfr_list):
        results.append({
            "title": item.get("title", f"NFR_{i+1}"),
            "description": item["description"],
            "predicted_type": pred_types[i],
            "predicted_level": pred_levels[i]
        })

    return results


# ============================================================
# 5️⃣ Save Predictions to MongoDB
# ============================================================

def save_predictions(results):
    collection = db["nfr_predictions_type_level"]
    collection.delete_many({})
    collection.insert_many(results)


# ============================================================
# 6️⃣ Main Pipeline
# ============================================================

def run_type_level_prediction():
    nfrs = load_extracted_nfrs()
    predictions = predict_type_and_level(nfrs)
    save_predictions(predictions)
    return predictions


# ============================================================
# 7️⃣ Local Testing
# ============================================================

if __name__ == "__main__":
    results = run_type_level_prediction()

    print("\n=== NFR TYPE + LEVEL PREDICTIONS ===")
    for r in results:
        print(
            f"{r['title']} → {r['predicted_type']} "
            f"(Level: {r['predicted_level']})"
        )
