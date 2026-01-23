"""
Train BERT models for:
1) NFR Type classification
2) NFR Level classification
"""

import pandas as pd
import numpy as np
import torch
import random
import os
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import evaluate
import inspect


# ============================================================
# 1️⃣ Reproducibility
# ============================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# 2️⃣ Paths & Config
# ============================================================

DATASET_PATH = "merged_NFR_cleaned_no_dots.csv"
MODEL_NAME = "bert-base-uncased"

SAVE_DIR_TYPE = "trained_nfr_type_model"
SAVE_DIR_LEVEL = "trained_nfr_level_model"

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5


# ============================================================
# 3️⃣ Load Dataset
# ============================================================

df = pd.read_csv(DATASET_PATH)

required_cols = {"Requirement", "Type", "Level"}
assert required_cols.issubset(df.columns), "Dataset columns missing"

print("Dataset sample:")
print(df.head())


# ============================================================
# 4️⃣ Encode Labels
# ============================================================

le_type = LabelEncoder()
le_level = LabelEncoder()

df["type_enc"] = le_type.fit_transform(df["Type"])
df["level_enc"] = le_level.fit_transform(df["Level"])


# ============================================================
# 5️⃣ Tokenizer
# ============================================================

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["Requirement"],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ============================================================
# 6️⃣ Metrics
# ============================================================

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }


# ============================================================
# 7️⃣ Training Arguments (shared)
# ============================================================

args = {
    "output_dir": "./results",
    "learning_rate": LR,
    "num_train_epochs": EPOCHS,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": BATCH_SIZE,
    "logging_dir": "./logs",
    "logging_steps": 20,
    "save_strategy": "epoch",
    "seed": SEED,
}

sig = inspect.signature(TrainingArguments.__init__)
if "evaluation_strategy" in sig.parameters:
    args["evaluation_strategy"] = "epoch"
if "do_eval" in sig.parameters:
    args["do_eval"] = True

training_args = TrainingArguments(**args)


# ============================================================
# 8️⃣ Train TYPE Model
# ============================================================

print("\n========== Training TYPE model ==========")

dataset_type = Dataset.from_pandas(
    df[["Requirement", "type_enc"]]
).rename_column("type_enc", "label")

splits_type = dataset_type.train_test_split(test_size=0.2, seed=SEED)

train_type = splits_type["train"].map(tokenize, batched=True)
eval_type = splits_type["test"].map(tokenize, batched=True)

train_type.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)
eval_type.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

model_type = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(le_type.classes_)
)

trainer_type = Trainer(
    model=model_type,
    args=training_args,
    train_dataset=train_type,
    eval_dataset=eval_type,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer_type.train()
trainer_type.evaluate()

os.makedirs(SAVE_DIR_TYPE, exist_ok=True)
model_type.save_pretrained(SAVE_DIR_TYPE)
tokenizer.save_pretrained(SAVE_DIR_TYPE)

print(f"✅ TYPE model saved to: {SAVE_DIR_TYPE}")


# ============================================================
# 9️⃣ Train LEVEL Model
# ============================================================

print("\n========== Training LEVEL model ==========")

dataset_level = Dataset.from_pandas(
    df[["Requirement", "level_enc"]]
).rename_column("level_enc", "label")

splits_level = dataset_level.train_test_split(test_size=0.2, seed=SEED)

train_level = splits_level["train"].map(tokenize, batched=True)
eval_level = splits_level["test"].map(tokenize, batched=True)

train_level.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)
eval_level.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

model_level = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(le_level.classes_)
)

trainer_level = Trainer(
    model=model_level,
    args=training_args,
    train_dataset=train_level,
    eval_dataset=eval_level,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer_level.train()
trainer_level.evaluate()

os.makedirs(SAVE_DIR_LEVEL, exist_ok=True)
model_level.save_pretrained(SAVE_DIR_LEVEL)
tokenizer.save_pretrained(SAVE_DIR_LEVEL)

print(f"✅ LEVEL model saved to: {SAVE_DIR_LEVEL}")
