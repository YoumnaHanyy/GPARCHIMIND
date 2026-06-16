# ai/training/train_binary.py

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate

from infrastructure.database import db   # 👈 Mongo connection

# =============================
# 1. Load dataset from MongoDB
# =============================
collection = db["merged_NFR_cleaned_no_dots"]

docs = list(
    collection.find(
        {},
        {
            "_id": 0,
            "Requirement": 1,
            "label": 1
        }
    )
)

df = pd.DataFrame(docs)

df = df.rename(columns={
    "Requirement": "text"
})

print("✅ Loaded binary training data from MongoDB")
print(df.head())

# =============================
# 2. Tokenization
# =============================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize, batched=True)

dataset = dataset.rename_column("label", "labels")
dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

dataset = dataset.train_test_split(test_size=0.2)

# =============================
# 3. Model
# =============================
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return metric.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="results_binary",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# =============================
# 4. Train & Save
# =============================
trainer.train()

model.save_pretrained("hanawahab/trained_nfr_binary_model")
tokenizer.save_pretrained("hanawahab/trained_nfr_binary_model")

print("✅ Binary NFR model trained & saved from MongoDB")