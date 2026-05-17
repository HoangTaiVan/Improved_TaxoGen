import os
import re
from collections import Counter


# ======================================================
# PATH CONFIG
# ======================================================

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

RAW_FILE = os.path.join(RAW_DIR, "papers_raw.txt")

PAPERS_FILE = os.path.join(PROCESSED_DIR, "papers.txt")
KEYWORDS_FILE = os.path.join(PROCESSED_DIR, "keywords.txt")
KEYWORD_CNT_FILE = os.path.join(PROCESSED_DIR, "keyword_cnt.txt")
INDEX_FILE = os.path.join(PROCESSED_DIR, "index.txt")


# ======================================================
# TEXT CLEANING
# ======================================================

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z_\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ======================================================
# KEYWORD EXTRACTION
# ======================================================

def extract_keywords(documents, min_freq=5, min_len=3):
    counter = Counter()

    for doc in documents:
        counter.update(doc.split())

    keywords = []

    for word, freq in counter.items():
        if len(word) < min_len:
            continue

        if freq < min_freq:
            continue

        keywords.append(word)

    return sorted(keywords)


# ======================================================
# SAVE FILES
# ======================================================

def save_papers(documents, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc + "\n")


def save_keywords(keywords, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for kw in keywords:
            f.write(kw + "\n")


def save_keyword_cnt(documents, keywords, output_file):
    keyword_set = set(keywords)

    with open(output_file, "w", encoding="utf-8") as f:
        for doc_id, doc in enumerate(documents):
            counter = Counter(doc.split())

            row = [str(doc_id)]

            for kw, cnt in counter.items():
                if kw in keyword_set:
                    row.append(kw)
                    row.append(str(cnt))

            f.write("\t".join(row) + "\n")


def save_index(documents, keywords, output_file):
    keyword_docs = {kw: [] for kw in keywords}

    for doc_id, doc in enumerate(documents):
        tokens = set(doc.split())

        for kw in keywords:
            if kw in tokens:
                keyword_docs[kw].append(str(doc_id))

    with open(output_file, "w", encoding="utf-8") as f:
        for kw, doc_ids in keyword_docs.items():
            f.write(f"{kw}\t{','.join(doc_ids)}\n")


# ======================================================
# MAIN
# ======================================================

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(
            f"Không tìm thấy file raw:\n{RAW_FILE}\n\n"
            f"Hãy tạo file papers_raw.txt trong thư mục data\\raw\\"
        )

    documents = []

    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line in f:
            clean = clean_text(line)

            if clean:
                documents.append(clean)

    print(f"[RAW] Documents: {len(documents)}")

    keywords = extract_keywords(
        documents,
        min_freq=5,
        min_len=3,
    )

    print(f"[RAW] Keywords: {len(keywords)}")

    save_papers(documents, PAPERS_FILE)
    save_keywords(keywords, KEYWORDS_FILE)
    save_keyword_cnt(documents, keywords, KEYWORD_CNT_FILE)
    save_index(documents, keywords, INDEX_FILE)

    print("[RAW] Saved:")
    print(f"  {PAPERS_FILE}")
    print(f"  {KEYWORDS_FILE}")
    print(f"  {KEYWORD_CNT_FILE}")
    print(f"  {INDEX_FILE}")


if __name__ == "__main__":
    main()