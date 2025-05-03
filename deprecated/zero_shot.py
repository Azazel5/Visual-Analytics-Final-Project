# bootstrap_labels.py

import json
import torch
from transformers import pipeline

# 1) Load pipelines
ner_pipe = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    tokenizer="dslim/bert-base-NER",
    aggregation_strategy="first",
    device=0 if torch.cuda.is_available() else -1
)

stance_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0 if torch.cuda.is_available() else -1
)

# 2) Read seed JSONL
seed = [json.loads(line) for line in open("zero_shot_sentiments_v2.jsonl", encoding="utf8")]

# 3) Annotate
bootstrapped = []
for record in seed:
    text = record["text"]
    # a) NER spans
    ents = ner_pipe(text)
    # normalize labels to match our tags
    spans = []
    for e in ents:
        tag = e["entity_group"]
        label = {
        "PER":  "PERSON",
        "ORG":  "ORG",
        "LOC":  "LOC",
        "EVENT": "EVENT" 
        }.get(tag)

        if not label:
            continue

        spans.append({"start": e["start"], "end": e["end"], "label": label})

    record.update({"spans": spans})
    bootstrapped.append(record)

# 4) Write out for review
with open("bootstrapped_labels_2.0.jsonl", "w", encoding="utf8") as f:
    for rec in bootstrapped:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Wrote {len(bootstrapped)} bootstrapped records")