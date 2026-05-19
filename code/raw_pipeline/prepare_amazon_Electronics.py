import os
import re
import sys
from collections import Counter

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.models.phrases import Phrases, Phraser


# ======================================================
# PATH
# ======================================================

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(ROOT_DIR)

from config.config import RAW_DIR, PROCESSED_DIR

META_FILE = os.path.join(RAW_DIR, "meta_Electronics.json")

PAPERS_FILE = os.path.join(PROCESSED_DIR, "papers.txt")
KEYWORDS_FILE = os.path.join(PROCESSED_DIR, "keywords.txt")


# ======================================================
# NLTK
# ======================================================

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# ======================================================
# FILTER CONFIG
# Mục tiêu: giữ vocabulary dưới 20k keywords
# ======================================================

MIN_DOC_TOKENS_RAW = 10
MIN_DOC_TOKENS_FINAL = 5

# Tăng min_df + idf để giảm keywords từ ~44k xuống khoảng 12k-18k
MIN_DF = 80
MAX_DF_RATIO = 0.20
MIN_IDF = 2.8

# Phrase mining chặt hơn để giảm phrase rác
PHRASE_MIN_COUNT = 40
PHRASE_THRESHOLD = 12


# ======================================================
# UTILS
# ======================================================

def list_to_text(x):
    if isinstance(x, list):
        out = []
        for item in x:
            if isinstance(item, list):
                out.extend([str(v) for v in item])
            else:
                out.append(str(item))
        return " ".join(out)

    if pd.isna(x):
        return ""

    return str(x)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ======================================================
# STOPWORDS
# ======================================================

base_stopwords = set(stopwords.words("english"))

custom_stopwords = {
    # platform / ecommerce noise
    "amazon", "product", "item", "seller", "shipping",
    "package", "pack", "piece", "pieces", "sale", "buy", "shop",
    "brand", "customer", "review", "seller", "vendor", "marketplace",

    # generic adjectives
    "new", "used", "good", "great", "nice", "perfect", "best",
    "easy", "high", "quality", "free", "cheap", "excellent",
    "amazing", "awesome", "wonderful", "beautiful", "better",

    # generic verbs / weak terms
    "use", "using", "make", "made", "include", "includes", "included",
    "provide", "provides", "designed", "desgin", "design", "allow", "allows",
    "come", "comes", "need", "needs", "want", "work", "works",

    # common ecommerce quantity / style noise
    "one", "two", "three", "inch", "inches", "foot", "feet",
    "black", "white", "blue", "red", "green", "yellow", "pink",
}

stop_words = base_stopwords.union(custom_stopwords)

lemmatizer = WordNetLemmatizer()


def clean_tokens(text):
    text = clean_text(text)
    tokens = []

    for w in text.split():
        if len(w) < 3:
            continue
        if len(w) > 30:
            continue
        if w in stop_words:
            continue
        if not w.isalpha():
            continue

        w = lemmatizer.lemmatize(w)

        if w in stop_words:
            continue

        tokens.append(w)

    return tokens


def is_bad_keyword(kw):
    """
    Lọc keyword/phrase nhiễu để taxonomy sạch hơn.
    """
    if not kw:
        return True

    if len(kw) <= 2 or len(kw) > 40:
        return True

    parts = kw.split("_")

    # Không giữ phrase quá dài
    if len(parts) > 3:
        return True

    # Bỏ phrase lặp kiểu surveillance_surveillance
    if len(parts) > 1 and len(set(parts)) == 1:
        return True

    # Bỏ phrase chứa token quá ngắn hoặc stopword
    for p in parts:
        if len(p) < 3:
            return True
        if p in stop_words:
            return True

    bad_words = {
        "electronics", "electronic", "product", "item", "device", "devices",
        "unit", "model", "compatible", "replacement", "accessory",
        "accessories", "generic", "original", "genuine", "unused",
        "condition", "description", "detail", "feature", "features",
        "price", "deal", "discount", "warranty", "guarantee",
        "available", "choice", "option", "variety", "color", "size",
    }

    if kw in bad_words:
        return True

    if any(p in bad_words for p in parts):
        return True

    # Bỏ từ có nhiều ký tự lạ dạng rác OCR/html còn sót
    if re.search(r"[^a-zA-Z_]+", kw):
        return True

    return False


# ======================================================
# MAIN
# ======================================================

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not os.path.exists(META_FILE):
        raise FileNotFoundError(f"Không thấy metadata file: {META_FILE}")

    print("[LOAD] Electronics metadata by chunks...")

    use_cols = [
        "asin",
        "title",
        "category",
        "feature",
        "description",
        "brand",
        "main_cat",
    ]

    product_docs = []

    chunk_iter = pd.read_json(
        META_FILE,
        lines=True,
        chunksize=50000,
    )

    for i, chunk in enumerate(chunk_iter):
        chunk = chunk[[c for c in use_cols if c in chunk.columns]].copy()

        for col in use_cols:
            if col not in chunk.columns:
                chunk[col] = ""

        for col in ["title", "category", "feature", "description", "brand", "main_cat"]:
            chunk[col] = chunk[col].apply(list_to_text)

        # Metadata only:
        # Giảm lặp so với bản cũ để tránh token bị nhân quá mạnh.
        # title/category vẫn là chính, nhưng không làm document quá nhiễu.
        chunk["full_text"] = (
            (chunk["title"].astype(str) + " ") * 3
            + (chunk["category"].astype(str) + " ") * 3
            + (chunk["feature"].astype(str) + " ") * 2
            + (chunk["brand"].astype(str) + " ") * 1
            + (chunk["main_cat"].astype(str) + " ") * 1
            + chunk["description"].astype(str)
        )

        chunk["tokens"] = chunk["full_text"].apply(clean_tokens)

        docs = [
            tokens
            for tokens in chunk["tokens"].tolist()
            if len(tokens) >= MIN_DOC_TOKENS_RAW
        ]

        product_docs.extend(docs)

        print(f"[CHUNK {i}] docs collected: {len(product_docs)}")

    print("[INFO] Product documents:", len(product_docs))
    if product_docs:
        print("[INFO] Sample product tokens:", product_docs[0][:40])

    # ======================================================
    # DOCUMENT FREQUENCY FILTER
    # ======================================================

    doc_freq = Counter()

    for words in product_docs:
        doc_freq.update(set(words))

    num_docs = len(product_docs)

    target_vocab = {
        w
        for w, dfreq in doc_freq.items()
        if (
            dfreq >= MIN_DF
            and dfreq <= MAX_DF_RATIO * num_docs
            and np.log(num_docs / (1 + dfreq)) > MIN_IDF
            and not is_bad_keyword(w)
        )
    }

    product_docs = [
        [w for w in words if w in target_vocab]
        for words in product_docs
    ]

    product_docs = [
        doc for doc in product_docs
        if len(doc) >= MIN_DOC_TOKENS_RAW
    ]

    print("[INFO] Target vocab size:", len(target_vocab))
    print("[INFO] Product docs after vocab filter:", len(product_docs))
    if product_docs:
        print("[INFO] Sample doc:", product_docs[0][:40])

    # ======================================================
    # PHRASE MINING
    # ======================================================

    phrases = Phrases(
        product_docs,
        min_count=PHRASE_MIN_COUNT,
        threshold=PHRASE_THRESHOLD,
    )

    bigram = Phraser(phrases)

    sentences = [
        bigram[doc]
        for doc in product_docs
    ]

    if sentences:
        print("[INFO] Example after phrase mining:", sentences[0][:40])

    # ======================================================
    # FINAL KEYWORDS
    # ======================================================

    final_counter = Counter()

    for sent in sentences:
        final_counter.update(set(sent))

    final_keywords = []

    for kw, freq in final_counter.items():
        if freq < MIN_DF:
            continue

        if is_bad_keyword(kw):
            continue

        # Ưu tiên phrase có nghĩa
        if "_" in kw:
            final_keywords.append(kw)

        # Giữ unigram kỹ thuật xuất hiện tốt
        elif kw in target_vocab:
            final_keywords.append(kw)

    # Sắp xếp theo tần suất giảm dần để giữ top keyword tốt nhất
    final_keywords = sorted(
        set(final_keywords),
        key=lambda x: (-final_counter[x], x),
    )


    # Sau khi cắt top, sort lại alphabet để output ổn định/dễ đọc
    final_keywords = sorted(final_keywords)
    keyword_set = set(final_keywords)

    print("[INFO] Final keywords:", len(final_keywords))
    print("[INFO] Sample keywords:", final_keywords[:50])

    # ======================================================
    # SAVE papers.txt
    # ======================================================

    valid_docs = []

    for sent in sentences:
        filtered = [
            w for w in sent
            if w in keyword_set
        ]

        if len(filtered) >= MIN_DOC_TOKENS_FINAL:
            valid_docs.append(filtered)

    with open(PAPERS_FILE, "w", encoding="utf-8") as f:
        for doc in valid_docs:
            f.write(" ".join(doc) + "\n")

    print("[SAVE] papers.txt:", len(valid_docs))

    # ======================================================
    # SAVE keywords.txt
    # ======================================================

    with open(KEYWORDS_FILE, "w", encoding="utf-8") as f:
        for kw in final_keywords:
            f.write(kw + "\n")

    print("[SAVE] keywords.txt:", len(final_keywords))

    print("[DONE] Electronics metadata preprocessing finished.")
    print(PAPERS_FILE)
    print(KEYWORDS_FILE)


if __name__ == "__main__":
    main()
