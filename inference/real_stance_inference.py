#!/usr/bin/env python3
import json
import random
from collections import defaultdict
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ──────────────────────────────────────────────────────────────────────────────
# 1) Configuration
# ──────────────────────────────────────────────────────────────────────────────

ORIGINAL_FILE = "bootstrapped_labels_2.0.jsonl"
GOLD_300_FILE = "phase1+2_gold.jsonl"
MODEL_DIR     = "stance-finetuned/"

# your three target labels:
LABELS       = ["NEG", "NEU", "POS"]
label2id     = {l: i for i, l in enumerate(LABELS)}
id2label     = {i: l for l, i in label2id.items()}


# ──────────────────────────────────────────────────────────────────────────────
# 2) Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def normalize_pred_label(raw: str) -> str:
    """Map pipeline outputs ('negative','neutral','positive') → our schema."""
    r = raw.lower()
    if r.startswith("neg"):
        return "NEG"
    if r.startswith("pos"):
        return "POS"
    return "NEU"


def load_gold_ids(path: str):
    """Return set of uid = 'filename|date' for the 300 examples."""
    s = set()
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            uid = f"{rec['metadata']['filename']}|{rec['metadata']['date']}"
            s.add(uid)
    return s


def load_unseen(original_path: str, gold_ids: set):
    """Filter out gold_ids from the full bootstrapped set."""
    out = []
    with open(original_path) as f:
        for line in f:
            rec = json.loads(line)
            uid = f"{rec['metadata']['filename']}|{rec['metadata']['date']}"
            if uid not in gold_ids:
                out.append(rec)
    return out


def split_sets(dataset, test_frac=0.1, dev_frac=0.1, seed=42):
    """
    Stratified split into dev/test/holdout.
    - test_frac & dev_frac are fractions of the total.
    - holdout_frac = 1 - (dev_frac + test_frac).
    """
    # Bucket by stance
    by_label = defaultdict(list)
    for rec in dataset:
        lbl = rec["stance"].replace("STANCE_", "")
        by_label[lbl].append(rec)

    dev_set, test_set, holdout = [], [], []
    # For each label, carve off dev/test/holdout
    for lbl, recs in by_label.items():
        # First take out dev+test as a pool
        pool_frac = dev_frac + test_frac
        pool, hold = train_test_split(
            recs,
            test_size=1 - pool_frac,
            random_state=seed,
            stratify=[lbl] * len(recs)
        )

        # Then split pool into dev vs test
        # relative fractions: dev_frac/(dev_frac+test_frac), test_frac/(dev_frac+test_frac)
        rel_test_size = test_frac / pool_frac
        d, t = train_test_split(
            pool,
            test_size=rel_test_size,
            random_state=seed,
            stratify=[lbl] * len(pool)
        )

        dev_set.extend(d)
        test_set.extend(t)
        holdout.extend(hold)

    # Shuffle to mix labels
    random.seed(seed)
    random.shuffle(dev_set)
    random.shuffle(test_set)
    random.shuffle(holdout)

    return dev_set, test_set, holdout

# ──────────────────────────────────────────────────────────────────────────────
# 3) Inference + evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_stance_and_eval(dataset, stance_pipe):
    # gold labels, stripped of the 'STANCE_' prefix
    gold = [rec["stance"].replace("STANCE_", "") for rec in dataset]
    texts = [rec["text"] for rec in dataset]

    # model returns list of lists of dicts if return_all_scores=True
    raw_outputs = stance_pipe(
        texts,
        batch_size=16,
        top_k=None,
    )

    # pick best for each and normalize
    preds = []
    for scores in raw_outputs:
        if not scores:
            preds.append("NEU")
            continue
        best = max(scores, key=lambda x: x["score"])
        preds.append(normalize_pred_label(best["label"]))

    # final classification report
    print(classification_report(
        gold,
        preds,
        labels=LABELS,
        zero_division=0,
        digits=4
    ))
    # also show accuracy
    acc = accuracy_score(gold, preds)
    print(f"Overall accuracy: {acc:.4f}\n")


# ──────────────────────────────────────────────────────────────────────────────
# 4) Main execution
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # load and split data
    gold_ids  = load_gold_ids(GOLD_300_FILE)
    all_unseen = load_unseen(ORIGINAL_FILE, gold_ids)
    dev_set, test_set, hold_set = split_sets(all_unseen)

    print(f"DEV size  : {len(dev_set)}   (≈10% of unseen)")
    print(f"TEST size : {len(test_set)}  (≈10% of unseen)")
    print(f"HOLD size : {len(hold_set)}  (≈80% of unseen)")

    # load tokenizer & model (ensure config has correct id2label/label2id)
    # if you need to re‑save your fine‑tuned model with proper config:
    # config = AutoConfig.from_pretrained(
    #     MODEL_DIR,
    #     num_labels=len(LABELS),
    #     id2label=id2label,
    #     label2id=label2id,
    # )
    # model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, config=config)

    # otherwise, assume MODEL_DIR already has a correct config.json
    stance_pipe = pipeline(
        "text-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        top_k=None,
        batch_size=16,
    )

    # evaluate
    print("\n=== DEV SET STANCE RESULTS ===")
    run_stance_and_eval(dev_set, stance_pipe)

    print("\n=== TEST SET STANCE RESULTS ===")
    run_stance_and_eval(test_set, stance_pipe)
