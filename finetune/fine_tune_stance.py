import numpy as np
import collections.abc
from sklearn.metrics import accuracy_score
from datasets import load_dataset, ClassLabel

# 3) pass dataset into Trainer as usual
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

DATA_PATH = "phase1+2_gold.jsonl"
LABELS    = ["NEU", "POS", "NEG"]
OUTPUT_DIR = "stance-finetuned"

# 1) define ClassLabel so we get .str2int mapping
class_label = ClassLabel(names=LABELS)
label2id     = {n: i for i, n in enumerate(class_label.names)}
tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest")

label_map = {"STANCE_NEG": 0, "STANCE_NEU": 1, "STANCE_POS": 2}


def flatten_and_clean(x):
    """Recursively flatten nested lists/tuples and keep only non-empty strings."""
    out = []
    if isinstance(x, str):
        return [x]
    if isinstance(x, collections.abc.Iterable):
        for y in x:
            out.extend(flatten_and_clean(y))
    return [s for s in out if isinstance(s, str) and s.strip()]

def preprocess(examples):
    # 1) Tokenize the entire batch of texts
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    # 2) Build a label list of the same length as the batch
    label_ids = []
    for stance_list in examples["stance"]:
        flat = flatten_and_clean(stance_list)
        if not flat:
            raise ValueError(f"No valid stance found in {stance_list}")
        
        first = flat[0]  # single-label classification: take the first
        label_ids.append(label_map[first])

    # 3) Attach to the batch dict
    tokenized["labels"] = label_ids
    return tokenized

def compute_accuracy(eval_pred):
    """
    eval_pred is a tuple (logits, labels) as returned by Trainer.predict/evaluate.
    logits.shape = (batch_size, num_labels)
    labels.shape = (batch_size,)
    """
    logits, labels = eval_pred
    # pick the highest logit as the predicted class
    preds = np.argmax(logits, axis=-1)

    # compute simple accuracy
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# 2) load & map
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.map(preprocess, remove_columns=dataset.column_names, batched=True)


model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    num_labels=len(LABELS),
)
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=100,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    eval_dataset=dataset, 
    tokenizer=tokenizer,
    compute_metrics=compute_accuracy,  
)

# 8) Train
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)