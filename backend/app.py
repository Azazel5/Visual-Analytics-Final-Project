# app.py
import json

from flask_cors import CORS
from flask import Flask, request, jsonify, abort

# ——————————————————————————————————————————
# 1) CONFIG
# ——————————————————————————————————————————
JSONL_PATH = "../bootstrapped_labels_2.0.jsonl"  
# or "bootstrapped_labels_2.0.jsonl", etc.

# Valid filter keys / allowed values
VALID_STANCES = {"STANCE_POS", "STANCE_NEG", "STANCE_NEU"}
VALID_ENTITY_LABELS = {"PER", "LOC", "ORG", "EVENT"}  # update to your schema

# ——————————————————————————————————————————
# 2) LOAD DATA
# ——————————————————————————————————————————
def load_predictions(path):
    preds = []
    with open(path, encoding="utf8") as f:
        for line in f:
            rec = json.loads(line)
            # sanity‐check shape:
            if "text" not in rec or "metadata" not in rec:
                continue
            preds.append(rec)
    return preds

all_predictions = load_predictions(JSONL_PATH)

# ——————————————————————————————————————————
# 3) FLASK APP
# ——————————————————————————————————————————
app = Flask(__name__)
CORS(app) 

@app.route("/predictions", methods=["GET"])
def get_predictions():
    """
    Query parameters:
      - source     : exact match on metadata.source
      - entities     : one of VALID_ENTITY_LABELS
      - stances     : one of VALID_STANCES
      - min_score  : float (0-1), default 0
      - limit      : int, max number of records to return (default 100)
    """
    src    = request.args.get("source", type=str)
    ent    = request.args.get("entities", type=str)
    stance = request.args.get("stances", type=str)
    min_sc = request.args.get("min_score", default=0.0, type=float)
    limit  = request.args.get("limit", default=100, type=int)
    ENTITY_NORMALIZER = {
    "PER":   "PERSON",
    "LOC":   "LOC",
    "ORG":   "ORG",
    "EVENT": "EVENT",
}

    # validate
    if ent and ent not in VALID_ENTITY_LABELS:
        abort(400, f"Unknown entity label: {ent}")
    if stance and stance not in VALID_STANCES:
        abort(400, f"Unknown stance: {stance}")

    # filter in‑memory
    results = []
    for rec in all_predictions:
        # 1) source filter
        if src and rec["metadata"].get("source") != src:
            continue

        # 2) stance filter
        if stance and rec.get("stance") != stance:
            continue

        # 3) score filter
        if rec.get("score", 0.0) < min_sc:
            continue

        # 4) entity filter: pass if any span.label matches
        if ent:
            if ent not in ENTITY_NORMALIZER:
                abort(400, f"Unknown entity label: {ent}")
            normalized_ent = ENTITY_NORMALIZER[ent]
        else:
            normalized_ent = None

        if normalized_ent:
            spans = rec.get("spans", [])
            if not any(s.get("label") == normalized_ent for s in spans):
                continue

        results.append(rec)
        if len(results) >= limit:
            break

    return jsonify(results)


@app.route("/", methods=["GET"])
def healthcheck():
    return "OK", 200


if __name__ == "__main__":
    # Use `flask run` or just: python app.py
    app.run(host="0.0.0.0", port=5005, debug=True)