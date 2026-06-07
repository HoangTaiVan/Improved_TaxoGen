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


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from config.config import RAW_DIR, PROCESSED_DIR


# ======================================================
# AUTO DETECT METADATA FILE
# ======================================================

META_CANDIDATES = [
    "meta_Amazon_Fashion.jsonl",
    "meta_AMAZON_FASHION.jsonl",
]

META_FILE = None

for fname in META_CANDIDATES:
    path = os.path.join(RAW_DIR, fname)
    if os.path.exists(path):
        META_FILE = path
        break

if META_FILE is None:
    raise FileNotFoundError(f"Không tìm thấy metadata file trong: {RAW_DIR}")


PAPERS_FILE = os.path.join(PROCESSED_DIR, "papers.txt")
KEYWORDS_FILE = os.path.join(PROCESSED_DIR, "keywords.txt")


nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# ======================================================
# CONFIG — STRICT FASHION ONLY
# ======================================================

MIN_DOC_TOKENS_RAW = 4
MIN_DOC_TOKENS_FINAL = 2

MIN_DF = 10
MAX_DF_RATIO = 0.30
MIN_IDF = 1.5

PHRASE_MIN_COUNT = 20
PHRASE_THRESHOLD = 10

TRIGRAM_MIN_COUNT = 10
TRIGRAM_THRESHOLD = 8

MAX_KEYWORDS = 12000


# ======================================================
# STRICT FASHION VOCAB
# ======================================================

STRICT_FASHION_KEEP = {
    "dress", "dresses", "shirt", "shirts", "tshirt", "tee", "top",
    "blouse", "sweater", "hoodie", "jacket", "coat", "cardigan",
    "skirt", "pants", "pant", "jeans", "jean", "shorts",
    "leggings", "legging", "suit", "gown", "romper", "jumpsuit",

    "shoe", "shoes", "sneaker", "sneakers", "boot", "boots",
    "sandal", "sandals", "heel", "heels", "flat", "flats",
    "slipper", "slippers", "loafer", "loafers", "pump", "pumps",

    "bag", "bags", "handbag", "purse", "wallet", "backpack",
    "tote", "clutch",

    "watch", "watches", "strap", "band",

    "jewelry", "jewellery", "necklace", "ring", "bracelet",
    "earring", "earrings", "pendant", "choker",

    "belt", "hat", "cap", "sock", "socks", "scarf", "glove", "gloves",

    "bra", "lingerie", "underwear", "swimsuit", "bikini",
    "swimwear", "activewear", "outerwear", "footwear", "clothing",
    "apparel", "accessory", "accessories",

    "cotton", "leather", "denim", "lace", "knit", "silk",
    "wool", "velvet", "suede", "mesh", "chiffon", "satin",
    "polyester", "spandex", "fleece",

    "sleeve", "collar", "waist", "zipper", "button", "pocket",
    "sole", "toe", "ankle", "hem", "buckle", "lining", "seam",

    "casual", "formal", "bridal", "wedding", "evening", "cocktail",
}

NON_FASHION_BLOCK = {
    "book", "books", "paperback", "hardcover", "textbook",
    "novel", "manga", "comic", "comics", "anime", "cartoon",
    "calendar", "poster", "dvd", "cd", "music", "movie", "film",
    "physics", "programming", "java", "statistics", "mechanics",
    "lecture", "lectures", "economy", "geography", "business",
    "coffee", "recipe", "recipes", "kitchen", "cookbook",
    "toy", "toys", "lego", "game", "games", "puzzle",
    "software", "electronics", "phone", "tablet", "camera",
    "lamp", "furniture", "tool", "tools", "equipment",
}

FASHION_DOMAIN_WORDS = STRICT_FASHION_KEEP


# ======================================================
# TEXT UTILS
# ======================================================

def list_to_text(x):
    if isinstance(x, list):
        out = []
        for item in x:
            if isinstance(item, list):
                out.extend([str(v) for v in item])
            elif isinstance(item, dict):
                out.extend([str(v) for v in item.values()])
            else:
                out.append(str(item))
        return " ".join(out)

    if isinstance(x, dict):
        return " ".join([str(v) for v in x.values()])

    if pd.isna(x):
        return ""

    return str(x)


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(
        r"\b("
        r"javascript|script|function|document|window|onclick|href|ajax|css|html|"
        r"amazonui|amznjq|signinredirect|addstyle|createelement|localstorage|"
        r"navigator|prototype|settimeout|console|cookie|browser|redirect|carousel|"
        r"buybox|popover|twister|swatch|coupon|navbar|sprite|asin"
        r")\b",
        " ",
        text,
    )

    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\b[a-z]{18,}\b", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_strict_fashion_text(text):
    text = clean_text(text)
    words = set(text.split())

    if len(words & NON_FASHION_BLOCK) > 0:
        return False

    return len(words & STRICT_FASHION_KEEP) >= 1


base_stopwords = set(stopwords.words("english"))

custom_stopwords = {
    "amazon", "product", "item", "seller", "shipping", "package",
    "pack", "piece", "pieces", "sale", "buy", "shop", "brand",
    "customer", "review", "vendor", "marketplace", "sold",
    "selling", "buyer", "purchase", "purchased", "received",
    "arrived", "order", "ordered",

    "new", "used", "good", "great", "nice", "perfect", "best",
    "easy", "high", "quality", "free", "cheap", "excellent",
    "amazing", "awesome", "beautiful", "cute", "pretty", "love",

    "one", "two", "three", "black", "white", "blue", "red",
    "green", "yellow", "pink", "color", "colors", "size",
    "small", "large", "medium",

    "gift", "christmas", "birthday", "present",

    "dimension", "weight", "pound", "inch", "inches", "approx",
    "approximately", "measure", "measurement", "may", "vary",
    "shown", "picture", "actual",

    "include", "includes", "included", "provide", "provides",
    "designed", "design", "come", "comes", "make", "made",
    "look", "looks", "looking", "use", "using",

    "main", "category", "categories", "detail", "details",
    "video", "videos", "image", "images", "store",
    "rating", "average", "number", "bought", "together",
}

stop_words = base_stopwords.union(custom_stopwords).union(NON_FASHION_BLOCK)
lemmatizer = WordNetLemmatizer()


BAD_SUBSTRINGS = [
    "javascript", "html", "css", "ajax", "href", "onclick",
    "amazonui", "amznjq", "carousel", "buybox", "popover",
    "twister", "swatch", "cookie", "navbar", "sprite",
    "dropdown", "checkbox", "radio", "button", "widget",
]


def clean_tokens(text):
    text = clean_text(text)
    tokens = []

    for w in text.split():
        if len(w) < 3 or len(w) > 30:
            continue
        if not w.isalpha():
            continue
        if w in stop_words:
            continue
        if any(s in w for s in BAD_SUBSTRINGS):
            continue

        w = lemmatizer.lemmatize(w)

        if w in stop_words:
            continue

        tokens.append(w)

    return tokens


def is_bad_token(w):
    if not w:
        return True
    if len(w) <= 2 or len(w) > 30:
        return True
    if w in stop_words:
        return True
    if w in NON_FASHION_BLOCK:
        return True
    if re.search(r"[^a-zA-Z]+", w):
        return True
    if any(s in w for s in BAD_SUBSTRINGS):
        return True

    bad_tokens = {
        "product", "item", "generic", "original", "genuine",
        "condition", "description", "detail", "feature", "features",
        "price", "deal", "discount", "warranty", "available",
        "choice", "option", "variety", "gift", "shipping",
        "package", "seller", "buyer", "imported", "machine",
        "wash", "dry", "hand", "main", "category", "categories",
        "average", "rating", "number", "store", "bought", "together",
    }

    return w in bad_tokens


def is_bad_keyword(kw):
    if not kw:
        return True
    if len(kw) <= 2 or len(kw) > 60:
        return True
    if re.search(r"[^a-zA-Z_]+", kw):
        return True
    if any(s in kw for s in BAD_SUBSTRINGS):
        return True

    parts = kw.split("_")

    if any(p in NON_FASHION_BLOCK for p in parts):
        return True

    if len(parts) == 1:
        return kw not in STRICT_FASHION_KEEP

    if len(parts) > 3:
        return True

    if len(set(parts)) == 1:
        return True

    for p in parts:
        if is_bad_token(p):
            return True

    if len(set(parts) & STRICT_FASHION_KEEP) == 0:
        return True

    return False


# ======================================================
# LOAD METADATA — STRICT FASHION ONLY
# ======================================================

def load_metadata_documents():
    print("[LOAD] Metadata only...")
    print("[LOAD META FILE]", META_FILE)

    if not os.path.exists(META_FILE):
        raise FileNotFoundError(f"Không thấy file: {META_FILE}")

    use_cols = [
        "parent_asin",
        "title",
        "store",
        "features",
        "description",
        "categories",
        "details",
        "main_category",
    ]

    docs = []

    for i, chunk in enumerate(pd.read_json(META_FILE, lines=True, chunksize=50000)):
        chunk = chunk[[c for c in use_cols if c in chunk.columns]].copy()

        for col in use_cols:
            if col not in chunk.columns:
                chunk[col] = ""

        for col in [
            "title",
            "store",
            "features",
            "description",
            "categories",
            "details",
            "main_category",
        ]:
            chunk[col] = chunk[col].apply(list_to_text)

        chunk["meta_text"] = (
            chunk["title"].astype(str) + " " +
            chunk["store"].astype(str) + " " +
            chunk["features"].astype(str) + " " +
            chunk["description"].astype(str) + " " +
            chunk["categories"].astype(str) + " " +
            chunk["details"].astype(str) + " " +
            chunk["main_category"].astype(str)
        )

        before = len(chunk)

        chunk = chunk[
            chunk["meta_text"].apply(is_strict_fashion_text)
        ].copy()

        print(f"[FILTER CHUNK {i}] strict fashion kept: {len(chunk)}/{before}")

        chunk = chunk.drop_duplicates(subset=["parent_asin"])
        chunk["tokens"] = chunk["meta_text"].apply(clean_tokens)

        chunk_docs = [
            tokens for tokens in chunk["tokens"].tolist()
            if len(tokens) >= MIN_DOC_TOKENS_RAW
            and len(set(tokens) & STRICT_FASHION_KEEP) >= 1
        ]

        docs.extend(chunk_docs)

        print(f"[META CHUNK {i}] docs collected: {len(docs)}")

    return docs


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    product_docs = load_metadata_documents()

    print("[INFO] Raw fashion documents:", len(product_docs))

    if product_docs:
        print("[INFO] Sample tokens:", product_docs[0][:40])

    doc_freq = Counter()
    for words in product_docs:
        doc_freq.update(set(words))

    num_docs = len(product_docs)

    target_vocab = {
        w for w, dfreq in doc_freq.items()
        if (
            dfreq >= MIN_DF
            and dfreq <= MAX_DF_RATIO * num_docs
            and np.log(num_docs / (1 + dfreq)) > MIN_IDF
            and not is_bad_token(w)
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
    print("[INFO] Docs after vocab filter:", len(product_docs))

    print("[PHRASE] Bigram...")
    bigram_phrases = Phrases(
        product_docs,
        min_count=PHRASE_MIN_COUNT,
        threshold=PHRASE_THRESHOLD
    )
    bigram = Phraser(bigram_phrases)
    bigram_docs = [bigram[doc] for doc in product_docs]

    print("[PHRASE] Trigram...")
    trigram_phrases = Phrases(
        bigram_docs,
        min_count=TRIGRAM_MIN_COUNT,
        threshold=TRIGRAM_THRESHOLD
    )
    trigram = Phraser(trigram_phrases)
    sentences = [trigram[doc] for doc in bigram_docs]

    if sentences:
        print("[INFO] Example after phrase mining:", sentences[0][:40])

    final_counter = Counter()
    for sent in sentences:
        final_counter.update(set(sent))

    final_keywords = []

    for kw, freq in final_counter.items():
        if freq < MIN_DF:
            continue

        if "_" not in kw and kw not in STRICT_FASHION_KEEP:
            continue

        if is_bad_keyword(kw):
            continue

        final_keywords.append(kw)

    final_keywords = sorted(
        set(final_keywords),
        key=lambda x: (-final_counter[x], x)
    )

    if MAX_KEYWORDS is not None:
        final_keywords = final_keywords[:MAX_KEYWORDS]

    final_keywords = sorted(final_keywords)
    keyword_set = set(final_keywords)

    print("[INFO] Final fashion keywords:", len(final_keywords))
    print("[INFO] Sample keywords:", final_keywords[:50])

    valid_docs = []

    for sent in sentences:
        filtered = [w for w in sent if w in keyword_set]
        if len(filtered) >= MIN_DOC_TOKENS_FINAL:
            valid_docs.append(filtered)

    print("[INFO] Valid fashion docs:", len(valid_docs))

    with open(PAPERS_FILE, "w", encoding="utf-8") as f:
        for doc in valid_docs:
            f.write(" ".join(doc) + "\n")

    print("[SAVE] papers.txt:", len(valid_docs))

    with open(KEYWORDS_FILE, "w", encoding="utf-8") as f:
        for kw in final_keywords:
            f.write(kw + "\n")

    print("[SAVE] keywords.txt:", len(final_keywords))

    print("[DONE] Strict fashion-only preprocessing finished.")
    print(PAPERS_FILE)
    print(KEYWORDS_FILE)


if __name__ == "__main__":
    main()