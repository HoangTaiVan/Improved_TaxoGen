import os
import json
import math
import pickle
import numpy as np
from collections import Counter


# ======================================================
# FILE / FOLDER UTILS
# ======================================================

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def ensure_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if parent:
        ensure_dir(parent)


def ensure_directory_exist(file_name):
    """
    Giữ lại tên hàm cũ để các file khác vẫn import được.
    """
    ensure_parent_dir(file_name)


def read_lines(file_path, encoding="utf-8"):
    with open(file_path, "r", encoding=encoding) as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(lines, file_path, encoding="utf-8"):
    ensure_parent_dir(file_path)

    with open(file_path, "w", encoding=encoding) as f:
        for line in lines:
            f.write(str(line) + "\n")


def save_json(obj, file_path):
    ensure_parent_dir(file_path)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj, file_path):
    ensure_parent_dir(file_path)

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


# ======================================================
# TEXT UTILS
# ======================================================

def tokenize(text):
    return str(text).lower().replace("_", " ").split()


def term_frequency(tokens):
    return Counter(tokens)


# ======================================================
# VECTOR UTILS
# ======================================================

def cossim(p, q, eps=1e-12):
    """
    Cosine similarity cho vector list hoặc numpy array.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    denom = np.linalg.norm(p) * np.linalg.norm(q)

    if denom < eps:
        return 0.0

    return float(np.dot(p, q) / denom)


def dot_product(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if len(p) != len(q):
        raise ValueError("dot_product error: p, q have different length")

    return float(np.dot(p, q))


def normalize_vector(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)

    if norm < eps:
        return v

    return v / norm


def normalize_matrix(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < eps] = 1.0

    return x / norms


# ======================================================
# OLD EMBEDDING TXT LOADER
# Giữ lại để đọc embedding.txt cũ nếu cần debug.
# Pipeline mới nên dùng .npy SBERT embeddings.
# ======================================================

def load_embeddings(embedding_file):
    if embedding_file is None:
        return {}

    word_to_vec = {}

    with open(embedding_file, "r", encoding="utf-8") as fin:
        first_line = fin.readline().strip().split()

        # Nếu dòng đầu là header dạng: vocab_size dim
        has_header = (
            len(first_line) == 2
            and first_line[0].isdigit()
            and first_line[1].isdigit()
        )

        if not has_header:
            word = first_line[0]
            vec = [float(v) for v in first_line[1:]]
            word_to_vec[word] = vec

        for line in fin:
            items = line.strip().split()

            if len(items) < 2:
                continue

            word = items[0]
            vec = [float(v) for v in items[1:]]
            word_to_vec[word] = vec

    return word_to_vec


# ======================================================
# SCORING UTILS
# ======================================================

def kl_divergence(p, q):
    if len(p) != len(q):
        raise ValueError("KL divergence error: p, q have different length")

    c_entropy = 0.0

    for i in range(len(p)):
        if p[i] > 0:
            c_entropy += p[i] * math.log(float(p[i]) / q[i])

    return c_entropy


def l1_normalize(p):
    sum_p = sum(p)

    if sum_p <= 0:
        return [0 for _ in p]

    return [x / sum_p for x in p]


def softmax(score_list):
    if not score_list:
        return []

    max_score = max(score_list)
    exp_list = [math.exp(score - max_score) for score in score_list]
    exp_sum = sum(exp_list)

    if exp_sum == 0:
        return [0 for _ in score_list]

    return [x / exp_sum for x in exp_list]


def softmax_for_map(t_map):
    if not t_map:
        return t_map

    keys = list(t_map.keys())
    values = [t_map[k] for k in keys]
    norm_values = softmax(values)

    for k, v in zip(keys, norm_values):
        t_map[k] = v

    return t_map


def minmax_normalize(scores):
    """
    Input:
        scores = {"keyword": score}
    Output:
        normalized score về [0, 1]
    """
    if not scores:
        return {}

    values = list(scores.values())
    mn = min(values)
    mx = max(values)

    if mx == mn:
        return {k: 1.0 for k in scores}

    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}


def bm25_score(tf, df, doc_len, avg_doc_len, n_docs, k1=1.5, b=0.75):
    """
    BM25 chuẩn dùng cho BM25 + Cosine ranker.
    """
    if tf <= 0 or df <= 0 or avg_doc_len <= 0:
        return 0.0

    idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    denom = tf + k1 * (1 - b + b * doc_len / avg_doc_len)

    if denom == 0:
        return 0.0

    return float(idf * ((tf * (k1 + 1)) / denom))


# ======================================================
# TAXONOMY / HIERARCHY UTILS
# ======================================================

def load_hier_f(hier_f):
    """
    Đọc hierarchy.txt dạng:
        child parent
    Output:
        {"child": cluster_index}
    """
    hier_map = {}

    with open(hier_f, "r", encoding="utf-8") as f:
        idx = 0

        for line in f:
            line = line.strip()

            if not line:
                continue

            topic = line.split()[0]
            hier_map[topic] = idx
            idx += 1

    return hier_map


# ======================================================
# COMPATIBILITY FUNCTIONS
# Các hàm cũ giữ lại để tránh lỗi import,
# nhưng pipeline mới hầu như không dùng.
# ======================================================

def euclidean_distance(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if len(p) != len(q):
        raise ValueError("Euclidean distance error: p, q have different length")

    return float(np.linalg.norm(p - q))


def euclidean_cluster(ps, c):
    if len(ps) == 0 or c is None:
        return 0.0

    ps = np.asarray(ps, dtype=float)
    c = np.asarray(c, dtype=float)

    distances = np.linalg.norm(ps - c, axis=1)
    return float(np.mean(distances))


def avg_emb(ele_map, embs_from, vec_size):
    avg = np.zeros(vec_size, dtype=float)
    total_weight = 0.0

    for key, value in ele_map.items():
        if key not in embs_from:
            continue

        avg += float(value) * np.asarray(embs_from[key], dtype=float)
        total_weight += float(value)

    if total_weight == 0:
        return avg.tolist()

    return (avg / total_weight).tolist()


def avg_emb_with_distinct(ele_map, embs_from, dist_map, vec_size):
    avg = np.zeros(vec_size, dtype=float)
    total_weight = 0.0

    for key, value in ele_map.items():
        if key not in embs_from:
            continue

        weight = float(value) * float(dist_map.get(key, 1.0))
        avg += weight * np.asarray(embs_from[key], dtype=float)
        total_weight += weight

    if total_weight == 0:
        return avg.tolist()

    return (avg / total_weight).tolist()


def avg_weighted_colors(color_list, c_size):
    result = np.zeros(c_size, dtype=float)

    for color, weight in color_list:
        result += float(weight) * np.asarray(color, dtype=float)

    return l1_normalize(result.tolist())

def load_txt_embeddings(embedding_file):

    word_to_vec = {}

    with open(embedding_file, "r", encoding="utf-8") as f:

        header = f.readline()

        for line in f:

            items = line.strip().split()

            if len(items) < 10:
                continue

            word = items[0]

            vec = np.array(
                [float(x) for x in items[1:]],
                dtype=np.float32,
            )

            word_to_vec[word] = vec

    return word_to_vec