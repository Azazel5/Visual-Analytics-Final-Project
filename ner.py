import json
from seqeval.metrics import classification_report
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer


def merge_doccano_jsonls():
    # load phase 1

    with open("spans_phase1_corrected.jsonl") as f1:
        spans = {rec["id"]: rec for rec in map(json.loads, f1)}

    # load phase 2
    with open("stance_phase2_corrected.jsonl") as f2:
        stances = {rec["id"]: rec for rec in map(json.loads, f2)}

    # merge
    merged = []
    for id_, rec in spans.items():
        if id_ not in stances:
            continue
        merged_rec = {
            "id": id_,
            "text": rec["text"],
            "metadata": rec["metadata"],
            "entities": rec["entities"],           # your corrected spans
            "stance": stances[id_]["stance"]       # your corrected stance
        }
        merged.append(merged_rec)

    # write out
    with open("phase1+2_gold.jsonl", "w") as out:
        for rec in merged:
            out.write(json.dumps(rec) + "\n")

# Function to convert gold spans to token-level BIO labels
def align_labels_to_tokens(entities, tokens, offsets):
    labels = ["O"] * len(tokens)
    for ent in entities:
        start, end, label = ent["start"], ent["end"], ent["label"]
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == start:
                labels[idx] = f"B-{label}"
            elif tok_start > start and tok_end <= end:
                labels[idx] = f"I-{label}"
    return labels

# Function to align predicted entities to BIO token labels
def preds_to_bio(preds, offsets, tokens):
    labels = ["O"] * len(offsets)
    for ent in preds:
        ent_label = ent["entity_group"]
        ent_start, ent_end = ent["start"], ent["end"]
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == ent_start:
                labels[i] = f"B-{ent_label}"
            elif tok_start > ent_start and tok_end <= ent_end:
                labels[i] = f"I-{ent_label}"
    return labels
    
def main():
    # 1. Load the combined gold dataset

    gold_data = []
    with open('phase1+2_gold.jsonl') as f:
        for line in f:
            gold_data.append(json.loads(line))

    print("Loaded gold data with {} records.".format(len(gold_data)))

    # 2. Initialize Hugging Face NER pipeline with a pre-trained model
    model_name = "dslim/bert-base-NER"  # you can swap to another model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    # 3. Prepare texts and gold labels
    texts = [item["text"] for item in gold_data]

    # Tokenize and produce gold_labels
    gold_labels = []
    tokenized_inputs = []
    offset_mappings = []

    print("Tokenizing and aligning labels...")
    
    for item in gold_data:
        toks = tokenizer(item["text"], return_offsets_mapping=True)
        offsets = toks["offset_mapping"]
        token_ids = toks["input_ids"]
        bio = align_labels_to_tokens(item["entities"], token_ids, offsets)
        gold_labels.append(bio)
        offset_mappings.append(offsets)

    # 4. Run NER inference
    print( "Running NER inference...")
    predictions = ner_pipeline(texts)

    pred_labels = []

    print("Aligning predictions to BIO labels...")
    
    for preds, offsets, toks in zip(predictions, offset_mappings, tokenized_inputs):
        bio = preds_to_bio(preds, offsets, toks["input_ids"])
        pred_labels.append(bio)

    # 5. Compute classification report
    print("Computing classification report...")
    report = classification_report(gold_labels, pred_labels)
    print("NER classification report on gold set:\n")
    print(report) 

if __name__ == "__main__":
    main()