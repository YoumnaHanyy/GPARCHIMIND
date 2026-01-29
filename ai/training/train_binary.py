"""
Train Binary NFR Classifier (BERT)
Label: 0 / 1
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import evaluate
import os
import inspect
import random


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

DATASET_PATH = "nfr_training_Binarydata.csv"
MODEL_NAME = "bert-base-uncased"
SAVE_DIR = "trained_nfr_binary_model"

NUM_LABELS = 2
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5


# ============================================================
# 3️⃣ Load Dataset
# ============================================================

df = pd.read_csv(DATASET_PATH)
df = df[["text", "label"]]   # enforce schema

print("Dataset sample:")
print(df.head())


# ============================================================
# 4️⃣ Tokenizer
# ============================================================

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )


# ============================================================
# 5️⃣ HuggingFace Dataset
# ============================================================

dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

splits = dataset.train_test_split(test_size=0.2, seed=SEED)
train_dataset = splits["train"]
eval_dataset = splits["test"]

print(f"Train size: {len(train_dataset)}")
print(f"Eval size: {len(eval_dataset)}")


# ============================================================
# 6️⃣ Model
# ============================================================

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)


# ============================================================
# 7️⃣ Metrics
# ============================================================

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(
        predictions=preds,
        references=labels
    )


# ============================================================
# 8️⃣ Training Arguments (Version Safe)
# ============================================================

args = {
    "output_dir": "./results_binary",
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
# 9️⃣ Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)


# ============================================================
# 🔟 Train
# ============================================================

trainer.train()

eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)


# ============================================================
# 1️⃣1️⃣ Save Model
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"✅ Binary model saved to: {SAVE_DIR}")
