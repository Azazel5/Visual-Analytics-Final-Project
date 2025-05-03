#!/usr/bin/env python3
import json
import random
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)
from seqeval.metrics import classification_report

ORIGINAL_FILE = "bootstrapped_labels_2.0.jsonl"
GOLD_300_FILE = "phase1+2_gold.jsonl"
MODEL_DIR     = "ner-finetuned/"

# -----------------------------------------------------------------------------
# 1) Read the 300 gold examples, remember their unique IDs
# -----------------------------------------------------------------------------
gold_ids = set()
with open(GOLD_300_FILE) as f:
    for line in f:
        rec = json.loads(line)
        uid = f"{rec['metadata']['filename']}|{rec['metadata']['date']}"
        gold_ids.add(uid)

# -----------------------------------------------------------------------------
# 2) Filter the original 1,200 down to the “unseen” set (~500)
# -----------------------------------------------------------------------------
remaining = []
with open(ORIGINAL_FILE) as f:
    for line in f:
        rec = json.loads(line)
        uid = f"{rec['metadata']['filename']}|{rec['metadata']['date']}"
        if uid not in gold_ids:
            remaining.append(rec)

print(f"Unseen examples (should be ~400): {len(remaining)}")

# -----------------------------------------------------------------------------
# 3) Split into DEV (10%), TEST (10%), HOLDOUT (80%)
# -----------------------------------------------------------------------------
random.seed(42)
random.shuffle(remaining)
dev_size  = int(0.1 * len(remaining))
test_size = dev_size

dev_set  = remaining[:dev_size]
test_set = remaining[dev_size : dev_size + test_size]

print(f"DEV size:  {len(dev_set)}")
print(f"TEST size: {len(test_set)}")

# -----------------------------------------------------------------------------
# 4) BIO‐conversion helpers
# -----------------------------------------------------------------------------
def normalize_label(lab: str) -> str:
    # map any source label into your schema
    if lab in ("PERSON",):
        return "PER"
    if lab in ("MISC",):
        # either treat as O or drop entirely
        return "O"
    return lab

def gold_to_bio_spans(text, spans, tokenizer):
    toks = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = toks["offset_mapping"]
    labels  = ["O"] * len(offsets)

    for span in spans:
        start, end = span["start"], span["end"]
        raw_lab    = span["label"]
        lab        = normalize_label(raw_lab)

        # skip if you want to treat MISC as O
        if lab == "O":
            continue

        for idx, (s, e) in enumerate(offsets):
            if s == start:
                labels[idx] = f"B-{lab}"
            elif s > start and e <= end:
                labels[idx] = f"I-{lab}"

    return labels, offsets

def ner_preds_to_bio(preds, offsets):
    labels = ["O"] * len(offsets)
    for ent in preds:
        raw_lab = ent["entity_group"]
        lab = normalize_label(raw_lab)
        if lab == "O":
            continue

        start, end = ent["start"], ent["end"]
        for i, (s, e) in enumerate(offsets):
            if s == start:
                labels[i] = f"B-{lab}"
            elif s > start and e <= end:
                labels[i] = f"I-{lab}"
    return labels

# -----------------------------------------------------------------------------
# 5) Run inference + evaluate
# -----------------------------------------------------------------------------
def run_ner_and_eval(dataset, ner_pipe, tokenizer):
    gold_seqs = []
    pred_seqs = []

    for rec in dataset:
        text  = rec["text"]
        spans = rec["spans"]       # your gold spans field

        # a) run model
        ann = ner_pipe(text)

        # b) align gold → BIO
        bio_gold, offsets = gold_to_bio_spans(text, spans, tokenizer)

        # c) align preds → BIO
        bio_pred = ner_preds_to_bio(ann, offsets)

        gold_seqs.append(bio_gold)
        pred_seqs.append(bio_pred)

    print(classification_report(gold_seqs, pred_seqs, zero_division=0))

# -----------------------------------------------------------------------------
# 6) Load model + tokenizer + pipeline
# -----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
ner_pipe  = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# -----------------------------------------------------------------------------
# 7) Eval on dev, then test
# -----------------------------------------------------------------------------
print("\n=== DEV SET RESULTS ===")
run_ner_and_eval(dev_set, ner_pipe, tokenizer)

print("\n=== TEST SET RESULTS ===")
run_ner_and_eval(test_set, ner_pipe, tokenizer)
