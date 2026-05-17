import os
import sys
import argparse
from collections import defaultdict

import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from common.utils import (
    read_lines,
    ensure_parent_dir,
    load_txt_embeddings,
)

from config.config import (
    MIN_CLUSTER_SIZE,
    MIN_SAMPLES,
    HDBSCAN_METRIC,
    CLUSTER_SELECTION_METHOD,
    CLUSTER_SELECTION_EPSILON,
)


def run_hdbscan(
    keyword_file,
    embedding_file,
    cluster_output_file,
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
):
    import hdbscan

    keywords = read_lines(keyword_file)

    embedding_map = load_txt_embeddings(embedding_file)

    keywords = [
        kw for kw in keywords
        if kw in embedding_map
    ]

    embeddings = np.array([
        embedding_map[kw]
        for kw in keywords
    ])

    if len(keywords) != len(embeddings):
        raise ValueError(
            f"Số keyword ({len(keywords)}) khác số embedding ({len(embeddings)})."
        )

    print("[HDBSCAN] Start clustering...")
    print(f"[HDBSCAN] keywords = {len(keywords)}")
    print(f"[HDBSCAN] min_cluster_size = {min_cluster_size}")
    print(f"[HDBSCAN] metric = {HDBSCAN_METRIC}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
    )

    labels = clusterer.fit_predict(embeddings)

    ensure_parent_dir(cluster_output_file)

    with open(cluster_output_file, "w", encoding="utf-8") as f:
        f.write("keyword\tcluster_id\n")

        for kw, label in zip(keywords, labels):
            f.write(f"{kw}\t{label}\n")

    cluster_map = defaultdict(list)

    for kw, label in zip(keywords, labels):
        cluster_map[int(label)].append(kw)

    valid_clusters = [
        c for c in cluster_map
        if c != -1
    ]

    print(f"[HDBSCAN] Number of clusters: {len(valid_clusters)}")
    print(f"[HDBSCAN] Noise keywords: {len(cluster_map.get(-1, []))}")
    print(f"[HDBSCAN] Saved: {cluster_output_file}")

    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--keyword_file", required=True)
    parser.add_argument("--embedding_file", required=True)
    parser.add_argument("--cluster_output_file", required=True)

    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=MIN_CLUSTER_SIZE,
    )

    parser.add_argument(
        "--min_samples",
        type=int,
        default=MIN_SAMPLES,
    )

    args = parser.parse_args()

    run_hdbscan(
        keyword_file=args.keyword_file,
        embedding_file=args.embedding_file,
        cluster_output_file=args.cluster_output_file,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )