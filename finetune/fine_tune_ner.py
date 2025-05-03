#!/usr/bin/env python3
import json
import datasets
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from seqeval.metrics import classification_report

# 1) Config
DATA_FILE = "phase1+2_gold.jsonl" 
LABEL_LIST = ["PER", "ORG", "LOC", "EVENT"]
MODEL_CHECKPOINT = "bert-base-cased"
OUTPUT_DIR       = "ner-finetuned"

label2id   = {label: i for i, label in enumerate(LABEL_LIST)}
id2label   = {i: label for label, i in label2id.items()}

# 2) Load dataset
def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)

raw = datasets.Dataset.from_list(list(read_jsonl(DATA_FILE)))
# split 90/10 for train/validation
split = raw.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split["train"], split["test"]

# 3) Tokenizer + alignment
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_and_align(ex):
    tokenized = tokenizer(
        ex["text"],
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )
    labels = []
    offsets = tokenized.pop("offset_mapping")
    for idx, (start, end) in enumerate(offsets):
        label = "O"
        for ent in ex["entities"]:
            s = ent.get("start", ent.get("start_offset"))
            e = ent.get("end",   ent.get("end_offset"))
            if start == s:
                label = f"B-{ent['label']}"
                break
            elif start >= s and end <= e:
                label = f"I-{ent['label']}"
                break
        labels.append(label2id[label.split("-",1)[-1]] if label!="O" else -100)
    tokenized["labels"] = labels
    return tokenized

train_ds = train_ds.map(tokenize_and_align, batched=False)
eval_ds  = eval_ds.map(tokenize_and_align,  batched=False)

# 4) Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# 5) Model
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(LABEL_LIST),
    id2label=id2label,
    label2id=label2id,
)

# 6) Metrics
def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=-1)
    true_labels = [[id2label[l] for l in seq if l != -100] 
                   for seq in labels]
    true_preds  = [
        [id2label[p] for (p, l) in zip(seq_pred, seq_lab) if l != -100]
        for seq_pred, seq_lab in zip(preds, labels)
    ]
    report = classification_report(true_labels, true_preds, zero_division=0)
    print(report)
    return {"f1": float(report.split()[-2])}

# 7) Trainer
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=100,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 8) Train
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
