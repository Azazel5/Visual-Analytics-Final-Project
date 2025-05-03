#!/usr/bin/env python3
import json
from transformers import pipeline, AutoTokenizer

# --- setup
zsp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["positive","negative","neutral"]
mapping = {"positive":"STANCE_POS","negative":"STANCE_NEG","neutral":"STANCE_NEU"}

# --- optional: ensemble templates
templates = [
    "This sentence expresses {} sentiment.",
    "Overall, the author is being {}.",
    "The tone of this text is {}."
]

def classify_ensemble(text):
    scores = {lbl:[] for lbl in candidate_labels}
    for tmpl in templates:
        out = zsp(text, candidate_labels, hypothesis_template=tmpl)
        for lbl, sc in zip(out["labels"], out["scores"]):
            scores[lbl].append(sc)
    avg = {lbl: sum(v)/len(v) for lbl,v in scores.items()}
    best = max(avg, key=avg.get)
    return mapping[best], avg[best]

# --- optional: chunking
tok = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
def classify_chunked(text, max_len=128, stride=64):
    toks = tok(text, return_overflowing_tokens=True,
               max_length=max_len, stride=stride, truncation=True)
    best_lbl, best_sc = None, -1.0
    for ids in toks["input_ids"]:
        seq = tok.decode(ids, skip_special_tokens=True)
        lbl, sc = classify_ensemble(seq)
        if sc > best_sc:
            best_lbl, best_sc = lbl, sc
    return best_lbl, best_sc

# --- run on unseen
IN = "doccano_seed.jsonl"
OUT= "zero_shot_sentiments_v2.jsonl"
with open(IN) as fin, open(OUT,"w") as fout:
    for line in fin:
        rec = json.loads(line)
        text = rec["text"]
        # pick one strategy:
        lbl, sc = classify_chunked(text)  # or classify_ensemble(text)
        rec["stance_zero_shot"] = lbl
        rec["zero_shot_score"]   = sc
        fout.write(json.dumps(rec)+"\n")

print("Wrote improved zero-shot labels →", OUT)