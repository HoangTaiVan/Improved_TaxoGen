import os
import sys
import argparse
import numpy as np

from collections import defaultdict, Counter

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from common.utils import (
    read_lines,
    ensure_parent_dir,
    cossim,
    minmax_normalize,
    load_txt_embeddings,
)

from config.config import (
    TOP_K,
    ALPHA_BM25,
    BETA_COSINE,
)


# ======================================================
# LOAD CLUSTERS
# ======================================================

def load_clusters(cluster_file):

    cluster_map = defaultdict(list)

    with open(cluster_file, "r", encoding="utf-8") as f:

        header = f.readline()

        for line in f:

            line = line.strip()

            if not line:
                continue

            keyword, cluster_id = line.split("\t")

            cluster_id = int(cluster_id)

            if cluster_id == -1:
                continue

            cluster_map[cluster_id].append(keyword)

    return cluster_map


# ======================================================
# LOAD DOCUMENTS
# ======================================================

def load_documents(document_file):

    docs = []

    with open(document_file, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line:
                docs.append(line.split())

    return docs


# ======================================================
# TF / DF
# ======================================================

def build_keyword_frequency(docs):

    keyword_tf = Counter()

    keyword_df = Counter()

    for doc in docs:

        keyword_tf.update(doc)

        for token in set(doc):
            keyword_df[token] += 1

    return keyword_tf, keyword_df


# ======================================================
# SIMPLE BM25
# ======================================================

def simple_bm25(
    keyword,
    keyword_tf,
    keyword_df,
    n_docs,
):
    tf = keyword_tf.get(keyword, 0)

    df = keyword_df.get(keyword, 0)

    if tf == 0 or df == 0:
        return 0.0

    k1 = 1.5
    b = 0.75

    idf = np.log(
        1 + (n_docs - df + 0.5) / (df + 0.5)
    )

    score = idf * (
        (tf * (k1 + 1))
        /
        (tf + k1 * (1 - b + b))
    )

    return float(score)


# ======================================================
# MAIN RANKER
# ======================================================

def rank_keywords(
    document_file,
    keyword_file,
    embedding_file,
    cluster_file,
    output_file,
    top_k=TOP_K,
    alpha=ALPHA_BM25,
    beta=BETA_COSINE,
):
    # ======================================================
    # LOAD DOCUMENTS
    # ======================================================

    docs = load_documents(document_file)

    keywords = read_lines(keyword_file)

    # ======================================================
    # LOAD TXT EMBEDDINGS
    # ======================================================

    embedding_map = load_txt_embeddings(
        embedding_file
    )

    keywords = [
        kw
        for kw in keywords
        if kw in embedding_map
    ]

    embeddings = np.array([
        embedding_map[kw]
        for kw in keywords
    ])

    keyword_to_id = {
        kw: i
        for i, kw in enumerate(keywords)
    }

    # ======================================================
    # LOAD CLUSTERS
    # ======================================================

    cluster_map = load_clusters(cluster_file)

    # ======================================================
    # TF / DF
    # ======================================================

    keyword_tf, keyword_df = (
        build_keyword_frequency(docs)
    )

    n_docs = len(docs)

    ensure_parent_dir(output_file)

    # ======================================================
    # OUTPUT
    # ======================================================

    with open(output_file, "w", encoding="utf-8") as f:

        f.write(
            "cluster_id\tkeyword\tbm25\t"
            "cosine\thybrid_score\n"
        )

        for cluster_id, cluster_keywords in cluster_map.items():

            cluster_vecs = []

            for kw in cluster_keywords:

                if kw in keyword_to_id:

                    cluster_vecs.append(
                        embeddings[keyword_to_id[kw]]
                    )

            if len(cluster_vecs) == 0:
                continue

            centroid = np.mean(
                cluster_vecs,
                axis=0,
            )

            bm25_scores = {}

            cosine_scores = {}

            # ======================================================
            # BM25 + COSINE
            # ======================================================

            for kw in cluster_keywords:

                if kw not in keyword_to_id:
                    continue

                bm25_scores[kw] = simple_bm25(
                    kw,
                    keyword_tf,
                    keyword_df,
                    n_docs,
                )

                cosine_scores[kw] = cossim(
                    embeddings[keyword_to_id[kw]],
                    centroid,
                )

            # ======================================================
            # NORMALIZE
            # ======================================================

            bm25_norm = minmax_normalize(
                bm25_scores
            )

            cosine_norm = minmax_normalize(
                cosine_scores
            )

            # ======================================================
            # HYBRID SCORE
            # ======================================================

            final_scores = {}

            for kw in cluster_keywords:

                final_scores[kw] = (
                    alpha * bm25_norm.get(kw, 0.0)
                    +
                    beta * cosine_norm.get(kw, 0.0)
                )

            ranked = sorted(
                final_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]

            # ======================================================
            # SAVE
            # ======================================================

            for kw, score in ranked:

                f.write(
                    f"{cluster_id}\t{kw}\t"
                    f"{bm25_scores.get(kw, 0.0):.6f}\t"
                    f"{cosine_scores.get(kw, 0.0):.6f}\t"
                    f"{score:.6f}\n"
                )

    print(f"[RANK] Saved: {output_file}")


# ======================================================
# CLI
# ======================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--document_file",
        required=True,
    )

    parser.add_argument(
        "--keyword_file",
        required=True,
    )

    parser.add_argument(
        "--embedding_file",
        required=True,
    )

    parser.add_argument(
        "--cluster_file",
        required=True,
    )

    parser.add_argument(
        "--output_file",
        required=True,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=TOP_K,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA_BM25,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=BETA_COSINE,
    )

    args = parser.parse_args()

    rank_keywords(
        document_file=args.document_file,
        keyword_file=args.keyword_file,
        embedding_file=args.embedding_file,
        cluster_file=args.cluster_file,
        output_file=args.output_file,
        top_k=args.top_k,
        alpha=args.alpha,
        beta=args.beta,
    )