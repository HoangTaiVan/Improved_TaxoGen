import os
import argparse
import queue
import numpy as np


# ======================================================
# UTILS
# ======================================================

def safe_read_lines(path):
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip()
        ]


def load_embeddings(embedding_file):
    embeddings = {}

    if not os.path.exists(embedding_file):
        return embeddings

    with open(embedding_file, "r", encoding="utf-8") as f:
        header = f.readline()

        for line in f:
            items = line.strip().split()

            if len(items) < 3:
                continue

            word = items[0]

            try:
                vec = np.array(
                    [float(x) for x in items[1:]],
                    dtype=np.float32,
                )
                embeddings[word] = vec
            except ValueError:
                continue

    return embeddings


def cosine(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def load_hierarchy(hierarchy_file):
    """
    hierarchy.txt format:
        child_folder parent_node
    """

    children = []

    if not os.path.exists(hierarchy_file):
        return children

    with open(hierarchy_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 1:
                continue

            child = parts[0]
            children.append(child)

    return children


def load_score_file(score_file):
    """
    keywords.txt-score.txt hoặc caseolap.txt:
        keyword<TAB>score
    """

    scores = {}

    if not os.path.exists(score_file):
        return scores

    with open(score_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")

            if len(parts) < 2:
                continue

            kw = parts[0]

            try:
                score = float(parts[1])
            except ValueError:
                score = 0.0

            scores[kw] = score

    return scores


# ======================================================
# REPRESENTATIVE PHRASES
# ======================================================

def get_representative_phrases(node_folder, node_label, top_n=10):
    """
    Chọn representative phrases cho node.

    Ưu tiên:
    1. keywords.txt-score.txt
    2. caseolap.txt
    3. keywords.txt
    4. embeddings.txt cosine với node label
    """

    result = []

    node_label = node_label.strip("/")

    if node_label and node_label != "*":
        label = node_label.split("/")[-1]
        result.append(label)

    # ======================================================
    # 1. keywords.txt-score.txt
    # ======================================================

    score_file = os.path.join(
        node_folder,
        "keywords.txt-score.txt",
    )

    scores = load_score_file(score_file)

    if len(scores) == 0:
        # ======================================================
        # 2. caseolap.txt
        # ======================================================

        caseolap_file = os.path.join(
            node_folder,
            "caseolap.txt",
        )

        scores = load_score_file(caseolap_file)

    if len(scores) > 0:
        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for kw, score in ranked:
            if kw not in result:
                result.append(kw)

            if len(result) >= top_n:
                return result[:top_n]

    # ======================================================
    # 3. keywords.txt
    # ======================================================

    keywords_file = os.path.join(
        node_folder,
        "keywords.txt",
    )

    keywords = safe_read_lines(keywords_file)

    for kw in keywords:
        if kw not in result:
            result.append(kw)

        if len(result) >= top_n:
            return result[:top_n]

    # ======================================================
    # 4. embeddings fallback
    # ======================================================

    emb_file = os.path.join(
        node_folder,
        "embeddings.txt",
    )

    embeddings = load_embeddings(emb_file)

    if len(embeddings) > 0:
        label = result[0] if result else None

        if label in embeddings:
            label_vec = embeddings[label]

            ranked = []

            for kw, vec in embeddings.items():
                ranked.append(
                    (kw, cosine(label_vec, vec))
                )

            ranked = sorted(
                ranked,
                key=lambda x: x[1],
                reverse=True,
            )

            for kw, score in ranked:
                if kw not in result:
                    result.append(kw)

                if len(result) >= top_n:
                    return result[:top_n]

    return result[:top_n]


# ======================================================
# COMPRESS TAXONOMY TREE
# ======================================================

def compress_taxonomy_tree(
    root_dir,
    output_file,
    top_n=10,
):
    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True,
    )

    q = queue.Queue()

    # current_folder, current_path
    q.put((root_dir, "*"))

    lines = []

    while not q.empty():
        current_folder, current_path = q.get()

        hierarchy_file = os.path.join(
            current_folder,
            "hierarchy.txt",
        )

        children = load_hierarchy(hierarchy_file)

        for child in children:
            child_folder = os.path.join(
                current_folder,
                child,
            )

            if current_path == "*":
                child_path = child
            else:
                child_path = f"{current_path}/{child}"

            q.put(
                (
                    child_folder,
                    child_path,
                )
            )

        if current_folder == root_dir:
            continue

        reps = get_representative_phrases(
            node_folder=current_folder,
            node_label=current_path,
            top_n=top_n,
        )

        reps_str = ",".join(reps)

        lines.append(
            f"{current_path}\t{reps_str}"
        )

    with open(output_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print("[DONE] Compressed taxonomy saved:")
    print(output_file)
    print(f"[INFO] Total nodes: {len(lines)}")


# ======================================================
# CLI
# ======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress Improved TaxoGen folder tree to taxonomy txt."
    )

    parser.add_argument(
        "-root",
        "--root",
        required=True,
        help="Root folder of taxogen_tree.",
    )

    parser.add_argument(
        "-output",
        "--output",
        required=True,
        help="Output taxonomy txt file.",
    )

    parser.add_argument(
        "-N",
        "--top_n",
        type=int,
        default=10,
        help="Number of representative phrases per node.",
    )

    args = parser.parse_args()

    compress_taxonomy_tree(
        root_dir=args.root,
        output_file=args.output,
        top_n=args.top_n,
    )