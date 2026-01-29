import numpy as np
import torch
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

from infrastructure.repositories.nfr_dataset_repository import NFRDatasetRepository



# ======================================
# 1️⃣ Load dataset from MongoDB
# ======================================
df = NFRDatasetRepository.load_nfr_dataset_from_mongo()

# لازم الأعمدة دي تكون موجودة:
# Requirement | Type | Level

le_type = LabelEncoder()
le_level = LabelEncoder()

df["type_enc"] = le_type.fit_transform(df["Type"])
df["level_enc"] = le_level.fit_transform(df["Level"])


# ======================================
# 2️⃣ Tokenizer
# ======================================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["Requirement"],
        padding=True,
        truncation=True,
        max_length=128
    )


# ======================================
# 3️⃣ Metrics
# ======================================
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }


# ======================================
# 4️⃣ Training args
# ======================================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch"
)

collator = DataCollatorWithPadding(tokenizer)


# ======================================
# 5️⃣ TYPE MODEL
# ======================================
dataset_type = Dataset.from_pandas(df[["Requirement", "type_enc"]])
dataset_type = dataset_type.rename_column("type_enc", "label")
dataset_type = dataset_type.train_test_split(test_size=0.2)

train_type = dataset_type["train"].map(tokenize, batched=True)
test_type = dataset_type["test"].map(tokenize, batched=True)

train_type.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_type.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model_type = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(le_type.classes_)
)

trainer_type = Trainer(
    model=model_type,
    args=training_args,
    train_dataset=train_type,
    eval_dataset=test_type,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics
)

trainer_type.train()

model_type.save_pretrained("models/trained_nfr_type_model")
tokenizer.save_pretrained("models/trained_nfr_type_model")


# ======================================
# 6️⃣ LEVEL MODEL
# ======================================
dataset_level = Dataset.from_pandas(df[["Requirement", "level_enc"]])
dataset_level = dataset_level.rename_column("level_enc", "label")
dataset_level = dataset_level.train_test_split(test_size=0.2)

train_level = dataset_level["train"].map(tokenize, batched=True)
test_level = dataset_level["test"].map(tokenize, batched=True)

train_level.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_level.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model_level = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(le_level.classes_)
)

trainer_level = Trainer(
    model=model_level,
    args=training_args,
    train_dataset=train_level,
    eval_dataset=test_level,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics
)

trainer_level.train()

model_level.save_pretrained("models/trained_nfr_level_model")
tokenizer.save_pretrained("models/trained_nfr_level_model")

print("✅ Training finished & models saved successfully")
