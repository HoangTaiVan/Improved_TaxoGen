import re
import numpy as np

from common.utils import cossim


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


def rank_cluster_keywords(cluster_keywords, keyword_to_id, embeddings, top_k=15):
    vecs = []

    for kw in cluster_keywords:
        if kw in keyword_to_id:
            vecs.append(embeddings[keyword_to_id[kw]])

    if len(vecs) == 0:
        return cluster_keywords[:top_k]

    centroid = np.mean(vecs, axis=0)

    scores = []

    for kw in cluster_keywords:
        if kw not in keyword_to_id:
            continue

        score = cossim(embeddings[keyword_to_id[kw]], centroid)
        scores.append((kw, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return [kw for kw, score in scores[:top_k]]


def choose_node_label(
    cluster_id,
    cluster_keywords,
    keyword_to_id,
    embeddings,
    used_labels=None,
):
    """
    Chọn tên node có nghĩa.
    Ưu tiên keyword gần centroid nhất.
    """

    if used_labels is None:
        used_labels = set()

    ranked = rank_cluster_keywords(
        cluster_keywords=cluster_keywords,
        keyword_to_id=keyword_to_id,
        embeddings=embeddings,
        top_k=20,
    )

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