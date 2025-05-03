import pandas as pd
import os, json, random, re
from nltk.tokenize import sent_tokenize

DATA_DIR       = "input_data/News Articles"      
MAX_ARTICLES   = 20           
MAX_SENTENCES  = 1200         
MANIFEST_CSV   = "manifest.csv"       

def clean_article(text):
    text = re.sub(r"<<\s*to continue reading.*?>>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def split_into_sentences(text):
    paras = text.split("\n\n")
    sents = []
    for para in paras:
        para = para.strip()
        if not para:
            continue
        for sent in sent_tokenize(para):
            sent = sent.strip()
            if sent:
                sents.append(sent)
    return sents

file_dates = {}
if MANIFEST_CSV:
    df = pd.read_csv(MANIFEST_CSV)  
    for _, row in df.iterrows():
        file_dates[(row.source, row.filename)] = row.date

seed_rows = []

for source in os.listdir(DATA_DIR):
    src_dir = os.path.join(DATA_DIR, source)
    if not os.path.isdir(src_dir):
        continue

    files = [f for f in os.listdir(src_dir) if f.endswith(".txt")]
    chosen = random.sample(files, min(MAX_ARTICLES, len(files)))

    for fname in chosen:
        path = os.path.join(src_dir, fname)
        raw = open(path, encoding="latin-1", errors="ignore").read()
        cleaned = clean_article(raw)
        sentences = split_into_sentences(cleaned)
        
        # 5. Assign a date if you have it, else use None
        date = file_dates.get((source, fname))

        # 6. Build one record per sentence
        for idx, sent in enumerate(sentences):
            seed_rows.append({
                "text": sent,
                "metadata": {
                    "source": source,
                    "filename": fname,
                    "date": date,
                    "sentence_index": idx
                }
            })

# Shuffle & trim
random.shuffle(seed_rows)
seed_rows = seed_rows[:MAX_SENTENCES]

# Write JSONL
with open("doccano_seed.jsonl", "w", encoding="utf8") as f:
    for row in seed_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"✅ Prepared {len(seed_rows)} sentences")
