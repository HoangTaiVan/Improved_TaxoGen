import re
import numpy as np

from common.utils import cossim


# ======================================================
# AMAZON FASHION PRIOR
# ======================================================

FASHION_PRIOR = {
    # clothing
    "shirt", "t_shirt", "tee", "dress", "skirt", "pants", "jeans",
    "jacket", "coat", "sweater", "hoodie", "blouse", "tunic",
    "romper", "jumpsuit", "gown", "suit", "polo", "shorts",
    "leggings", "legging", "cardigan", "vest",

    # footwear
    "shoe", "shoes", "sneaker", "sneakers", "boot", "boots",
    "sandal", "sandals", "heel", "heels", "flat", "flats",
    "slipper", "loafer", "wedge", "pump",

    # bags / accessories
    "bag", "handbag", "purse", "wallet", "backpack", "tote",
    "clutch", "watch", "ring", "necklace", "bracelet",
    "earring", "earrings", "pendant", "choker", "belt",
    "scarf", "hat", "cap", "glove", "gloves",

    # underwear / swimwear
    "bra", "lingerie", "underwear", "swimsuit", "bikini",
    "swimwear", "tankini",

    # materials / style
    "cotton", "leather", "denim", "lace", "knit", "silk",
    "wool", "suede", "velvet", "satin", "chiffon", "polyester",
    "spandex", "fleece", "mesh",

    # parts
    "sleeve", "collar", "waist", "strap", "zipper", "button",
    "pocket", "sole", "toe", "ankle", "heel", "buckle",
    "lining", "seam",

    # occasions
    "casual", "formal", "bridal", "wedding", "evening",
    "cocktail", "activewear", "outerwear",
}


BAD_LABEL_TOKENS = {
    # review noise
    "good", "great", "nice", "perfect", "best", "better",
    "excellent", "awesome", "amazing", "beautiful", "cute",
    "pretty", "love", "loved", "like", "liked", "recommend",
    "recommended", "happy", "disappointed",

    # shopping noise
    "product", "item", "seller", "buyer", "customer", "shipping",
    "package", "order", "ordered", "received", "arrived",
    "purchase", "purchased", "price", "deal", "discount",
    "sale", "cheap", "expensive", "refund", "return", "returned",

    # generic
    "new", "used", "quality", "option", "choice", "variety",
    "feature", "features", "description", "detail", "details",
    "size", "small", "large", "medium", "color", "colors",

    # gift / seasonal noise
    "gift", "christmas", "birthday", "present",

    # web/html noise
    "amazon", "asin", "html", "css", "javascript", "script",
    "button", "image", "url", "href", "onclick", "ajax",
}


BRAND_NOISE_HINTS = {
    "caribbean", "joe", "koloa", "surf", "mix", "trendz",
    "allegra", "floerns", "zmart", "nike", "puma", "lego",
    "disney", "adidas", "reebok", "calvin", "klein",
}


def make_safe_node_name(name):
    name = str(name).strip().lower()
    name = name.replace(" ", "_")
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace(":", "_")
    name = name.replace("*", "root")
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def keyword_parts(keyword):
    return [
        p for p in str(keyword).lower().split("_")
        if p
    ]


def is_noisy_label(keyword):
    if not keyword:
        return True

    keyword = make_safe_node_name(keyword)

    if len(keyword) < 3 or len(keyword) > 60:
        return True

    parts = keyword_parts(keyword)

    if len(parts) < 1 or len(parts) > 4:
        return True

    if len(set(parts)) == 1:
        return True

    if any(len(p) > 18 for p in parts):
        return True

    if len(set(parts) & BAD_LABEL_TOKENS) > 0:
        return True

    # Nếu toàn là brand / tên riêng mà không có từ fashion thì loại
    has_fashion = len(set(parts) & FASHION_PRIOR) > 0
    has_brand_noise = len(set(parts) & BRAND_NOISE_HINTS) > 0

    if has_brand_noise and not has_fashion:
        return True

    return False


def fashion_score(keyword):
    parts = keyword_parts(keyword)
    kw = make_safe_node_name(keyword)

    score = 0.0

    hit = len(set(parts) & FASHION_PRIOR)

    if kw in FASHION_PRIOR:
        score += 0.50

    if hit > 0:
        score += 0.35

    if hit >= 2:
        score += 0.25

    # Label 2-3 từ thường đẹp hơn:
    # leather_bag, ankle_boot, floral_dress
    if len(parts) == 2:
        score += 0.15

    if len(parts) == 3:
        score += 0.08

    if len(parts) > 3:
        score -= 0.15

    return score


def noise_penalty(keyword):
    parts = keyword_parts(keyword)

    penalty = 0.0

    if len(parts) >= 4:
        penalty += 0.15

    if any(len(p) > 16 for p in parts):
        penalty += 0.20

    if len(set(parts) & BAD_LABEL_TOKENS) > 0:
        penalty += 0.60

    has_fashion = len(set(parts) & FASHION_PRIOR) > 0
    has_brand_noise = len(set(parts) & BRAND_NOISE_HINTS) > 0

    if has_brand_noise and not has_fashion:
        penalty += 0.50

    return penalty


def rank_cluster_keywords(
    cluster_keywords,
    keyword_to_id,
    embeddings,
    keyword_freq=None,
    top_k=15,
):
    valid_keywords = [
        kw for kw in cluster_keywords
        if kw in keyword_to_id
    ]

    if len(valid_keywords) == 0:
        return cluster_keywords[:top_k]

    vecs = np.array([
        embeddings[keyword_to_id[kw]]
        for kw in valid_keywords
    ])

    centroid = np.mean(vecs, axis=0)

    scores = []

    for kw in valid_keywords:
        emb = embeddings[keyword_to_id[kw]]
        cosine_score = cossim(emb, centroid)

        freq_bonus = 0.0
        if keyword_freq is not None:
            freq_bonus = 0.04 * np.log(
                1.0 + keyword_freq.get(kw, 1)
            )

        final_score = (
            cosine_score
            + fashion_score(kw)
            + freq_bonus
            - noise_penalty(kw)
        )

        if is_noisy_label(kw):
            final_score -= 0.8

        scores.append((kw, final_score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return [kw for kw, score in scores[:top_k]]


def choose_node_label(
    cluster_id,
    cluster_keywords,
    keyword_to_id,
    embeddings,
    used_labels=None,
    keyword_freq=None,
):
    """
    Chọn label cho Amazon Fashion taxonomy.

    Ưu tiên:
    1. keyword gần centroid cụm
    2. có từ thuộc miền fashion
    3. không phải review/shopping/html noise
    4. không trùng label đã dùng
    """

    if used_labels is None:
        used_labels = set()

    ranked = rank_cluster_keywords(
        cluster_keywords=cluster_keywords,
        keyword_to_id=keyword_to_id,
        embeddings=embeddings,
        keyword_freq=keyword_freq,
        top_k=30,
    )

    # vòng 1: label sạch + có fashion prior
    for kw in ranked:
        label = make_safe_node_name(kw)

        if not label:
            continue

        if is_noisy_label(label):
            continue

        parts = keyword_parts(label)

        if label in FASHION_PRIOR or len(set(parts) & FASHION_PRIOR) > 0:
            if label not in used_labels:
                used_labels.add(label)
                return label, ranked

    # vòng 2: label sạch bất kỳ
    for kw in ranked:
        label = make_safe_node_name(kw)

        if not label:
            continue

        if is_noisy_label(label):
            continue

        if label not in used_labels:
            used_labels.add(label)
            return label, ranked

    # vòng 3: fallback
    for kw in ranked:
        label = make_safe_node_name(kw)

        if label and label not in used_labels:
            used_labels.add(label)
            return label, ranked

    fallback = f"cluster_{cluster_id}"

    while fallback in used_labels:
        fallback = fallback + "_x"

    used_labels.add(fallback)

    return fallback, ranked