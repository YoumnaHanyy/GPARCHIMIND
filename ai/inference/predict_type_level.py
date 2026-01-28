import json
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from ai.utils.nfr_mapping import NFR_MAP_REVERSE

# load dataset from MongoDB (for encoders)
from infrastructure.repositories.nfr_dataset_repository import (
    NFRDatasetRepository,
    NFRPredictionRepository
)


BASE_OUTPUT = "data/outputs"

NFR_INPUT_PATH = os.path.join(BASE_OUTPUT, "non_functional_requirements.json")
OUTPUT_PATH = os.path.join(BASE_OUTPUT, "nfr_predictions_type_level.json")

MODEL_TYPE_PATH = "models/trained_nfr_type_model"
MODEL_LEVEL_PATH = "models/trained_nfr_level_model"


def predict_and_save_nfr():
    """
    Reads extracted NFRs from JSON,
    predicts Type + Level,
    saves results to JSON,
    and returns predictions list.
    """

    # ===============================
    # 1️⃣ Load dataset from MongoDB (for LabelEncoders)
    # ===============================
    df = NFRDatasetRepository.load_nfr_dataset_from_mongo()

    le_type = LabelEncoder()
    le_type.fit(df["Type"])

    le_level = LabelEncoder()
    le_level.fit(df["Level"])


    # ===============================
    # 2️⃣ Load tokenizer & models
    # ===============================
    tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE_PATH)

    model_type = BertForSequenceClassification.from_pretrained(MODEL_TYPE_PATH)
    model_level = BertForSequenceClassification.from_pretrained(MODEL_LEVEL_PATH)

    model_type.eval()
    model_level.eval()


    # ===============================
    # 3️⃣ Load extracted NFRs
    # ===============================
    with open(NFR_INPUT_PATH, "r", encoding="utf-8") as f:
        nfrs = json.load(f)

    texts = [item["description"] for item in nfrs]

    if not texts:
        raise ValueError("❌ No NFRs found to predict")


    # ===============================
    # 4️⃣ Tokenize
    # ===============================
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )


    # ===============================
    # 5️⃣ Predict Type + Level
    # ===============================
    with torch.no_grad():
        logits_type = model_type(**tokens).logits
        logits_level = model_level(**tokens).logits

    pred_type_ids = torch.argmax(logits_type, dim=1).cpu().numpy()
    pred_level_ids = torch.argmax(logits_level, dim=1).cpu().numpy()

    pred_types = le_type.inverse_transform(pred_type_ids)
    pred_levels = le_level.inverse_transform(pred_level_ids)


    # ===============================
    # 6️⃣ Build output JSON
    # ===============================
    results = []
    for i, item in enumerate(nfrs):

        pred_type_abbrev = pred_types[i]

        results.append({
    "title": item.get("title"),
    "description": item.get("description"),

    # 🔒 for DB / logic
    "predicted_type": pred_type_abbrev,

    # 👀 for UI only
    "predicted_type_label": NFR_MAP_REVERSE.get(
        pred_type_abbrev, pred_type_abbrev
    ),

    "predicted_level": pred_levels[i]
})




    # ===============================
    # 7️⃣ Save results
    # ===============================
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ NFR predictions saved to: {OUTPUT_PATH}")
    
    NFRPredictionRepository.save(project_id=2, predictions=results)
    return results
