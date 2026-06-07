import os
import csv
import argparse
from collections import defaultdict, Counter

# ============================================================
# Fleiss' Kappa Evaluation for TaxoGen Human Evaluation
#
# Input folder cần có:
#   intrusion_gold.txt
#   subdomain_gold.txt
#   *_intrusion_labels.csv
#   *_subdomain_labels.csv
#
# Chạy:
#   py kappa_eval_final_fixed.py -folder "D:\DATN\impoved_taxogen\code\data\result_dblp\eval_output_dblp"
#
# Chỉ tính từng task:
#   py kappa_eval_final_fixed.py -folder "..." -task intrusion
#   py kappa_eval_final_fixed.py -folder "..." -task subdomain
# ============================================================

YES_VALUES = {"y", "yes", "1", "true", "đúng", "dung", "có", "co"}
NO_VALUES = {"n", "no", "0", "false", "sai", "không", "khong"}


# ============================================================
# CSV helpers
# ============================================================

def open_csv_dict(path):
    """
    Mở file CSV, tự nhận diện dấu phân cách ',' hoặc ';'.
    """
    f = open(path, "r", encoding="utf-8-sig", newline="")
    sample = f.read(4096)
    f.seek(0)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        reader = csv.DictReader(f, dialect=dialect)
    except csv.Error:
        reader = csv.DictReader(f)

    return f, reader


def normalize_header(row):
    """
    Chuẩn hóa key/value trong một dòng CSV.
    """
    return {
        str(k).strip(): "" if v is None else str(v).strip()
        for k, v in row.items()
        if k is not None
    }


def read_gold_by_id(gold_file):
    """
    Đọc file gold của TaxoGen.

    Lưu ý:
    Tool web/eval của bạn đang dùng ID hiển thị bắt đầu từ 2,
    nên enumerate(start=2) để khớp intrusion_id/subdomain_id.
    """
    gold = {}

    with open(gold_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=2):
            segs = line.strip("\r\n").split("\t")

            if len(segs) < 3:
                continue

            gold[str(idx)] = {
                "key": segs[0].strip(),
                "method": segs[1].strip(),
                "gold": segs[2].strip()
            }

    return gold


# ============================================================
# Fix broken rows from Excel
# ============================================================

def fix_broken_intrusion_row(row):
    """
    Sửa trường hợp Excel bọc nguyên dòng intrusion CSV vào 1 ô.
    Format kỳ vọng:
        user_email,intrusion_id,word_1,word_2,word_3,word_4,word_5,word_6,outlier_id,labeled_at
    """
    if row is None:
        return row

    keys = list(row.keys())

    if (
        "intrusion_id" in keys and row.get("intrusion_id")
        and "outlier_id" in keys and row.get("outlier_id")
    ):
        return row

    candidates = []

    for k, v in row.items():
        if k is not None:
            candidates.append(str(k))
        if v is not None:
            candidates.append(str(v))

    for raw in candidates:
        if "intrusion_id" in raw and "outlier_id" in raw:
            continue

        if raw.count(",") >= 9:
            try:
                parsed = next(csv.reader([raw]))
            except Exception:
                continue

            if len(parsed) >= 10:
                return {
                    "user_email": parsed[0],
                    "intrusion_id": parsed[1],
                    "word_1": parsed[2],
                    "word_2": parsed[3],
                    "word_3": parsed[4],
                    "word_4": parsed[5],
                    "word_5": parsed[6],
                    "word_6": parsed[7],
                    "outlier_id": parsed[8],
                    "labeled_at": parsed[9],
                }

    return row


def fix_broken_subdomain_row(row):
    """
    Sửa trường hợp Excel bọc nguyên dòng subdomain CSV vào 1 ô.
    Format kỳ vọng:
        user_email,subdomain_id,child,parent,label,labeled_at
    """
    if row is None:
        return row

    keys = list(row.keys())

    if (
        "subdomain_id" in keys and row.get("subdomain_id")
        and "label" in keys and row.get("label")
    ):
        return row

    candidates = []

    for k, v in row.items():
        if k is not None:
            candidates.append(str(k))
        if v is not None:
            candidates.append(str(v))

    for raw in candidates:
        if "subdomain_id" in raw and "label" in raw:
            continue

        if raw.count(",") >= 5:
            try:
                parsed = next(csv.reader([raw]))
            except Exception:
                continue

            if len(parsed) >= 6:
                return {
                    "user_email": parsed[0],
                    "subdomain_id": parsed[1],
                    "child": parsed[2],
                    "parent": parsed[3],
                    "label": parsed[4],
                    "labeled_at": parsed[5],
                }

    return row


# ============================================================
# Label normalization
# ============================================================

def normalize_intrusion_label(pred):
    """
    Chuẩn hóa nhãn Topic Intrusion về dạng 0..5.

    Web thường lưu outlier_id = 1..6.
    Một số file có thể lưu 0..5.
    """
    pred = str(pred).strip()

    if not pred.isdigit():
        return None

    value = int(pred)

    # Nếu UI lưu 1..6, đổi về 0..5
    if 1 <= value <= 6:
        return str(value - 1)

    # Nếu đã là 0..5 thì giữ nguyên
    if 0 <= value <= 5:
        return str(value)

    return None


def normalize_subdomain_label(value):
    """
    Chuẩn hóa nhãn Parent-Child về y/n.
    """
    value = str(value).strip().lower()

    if value in YES_VALUES:
        return "y"

    if value in NO_VALUES:
        return "n"

    return None


# ============================================================
# Fleiss' Kappa
# ============================================================

def fleiss_kappa_from_matrix(matrix):
    """
    Tính Fleiss' Kappa từ ma trận đếm.

    Ví dụ Parent-Child với 5 annotators:
        [5, 0] = cả 5 người chọn y
        [4, 1] = 4 người chọn y, 1 người chọn n

    Formula:
        kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    """
    if not matrix:
        return None

    n_items = len(matrix)
    n_raters = sum(matrix[0])

    if n_items == 0 or n_raters <= 1:
        return None

    for row in matrix:
        if sum(row) != n_raters:
            raise ValueError(
                "Mỗi item phải có cùng số annotators. "
                "Có thể một số câu bị thiếu nhãn từ một annotator."
            )

    # P_i: mức đồng thuận trên từng item
    p_i_values = []
    for row in matrix:
        numerator = sum(n_ij * (n_ij - 1) for n_ij in row)
        denominator = n_raters * (n_raters - 1)
        p_i_values.append(numerator / denominator)

    p_bar = sum(p_i_values) / n_items

    # p_j: tỷ lệ toàn cục của từng nhãn
    n_categories = len(matrix[0])
    p_j_values = []

    for j in range(n_categories):
        total_j = sum(row[j] for row in matrix)
        p_j_values.append(total_j / (n_items * n_raters))

    p_e_bar = sum(p_j ** 2 for p_j in p_j_values)

    if abs(1 - p_e_bar) < 1e-12:
        return 1.0

    return (p_bar - p_e_bar) / (1 - p_e_bar)


def interpret_kappa(kappa):
    """
    Diễn giải theo thang Landis & Koch thường dùng.
    """
    if kappa is None:
        return "N/A"
    if kappa < 0:
        return "Poor agreement"
    if kappa <= 0.20:
        return "Slight agreement"
    if kappa <= 0.40:
        return "Fair agreement"
    if kappa <= 0.60:
        return "Moderate agreement"
    if kappa <= 0.80:
        return "Substantial agreement"
    return "Almost perfect agreement"


# ============================================================
# Load labels
# ============================================================

def load_intrusion_labels(folder):
    labels_by_item = defaultdict(dict)

    files = sorted([
        f for f in os.listdir(folder)
        if f.endswith("_intrusion_labels.csv")
    ])

    for fname in files:
        path = os.path.join(folder, fname)
        f, reader = open_csv_dict(path)

        with f:
            for raw_row in reader:
                raw_row = fix_broken_intrusion_row(raw_row)
                row = normalize_header(raw_row)

                item_id = row.get("intrusion_id", "").strip()
                pred = row.get("outlier_id", "").strip()

                if not item_id or not pred:
                    continue

                label = normalize_intrusion_label(pred)

                if label is None:
                    continue

                labels_by_item[item_id][fname] = label

    return labels_by_item, files


def load_subdomain_labels(folder):
    labels_by_item = defaultdict(dict)

    files = sorted([
        f for f in os.listdir(folder)
        if f.endswith("_subdomain_labels.csv")
    ])

    for fname in files:
        path = os.path.join(folder, fname)
        f, reader = open_csv_dict(path)

        with f:
            for raw_row in reader:
                raw_row = fix_broken_subdomain_row(raw_row)
                row = normalize_header(raw_row)

                item_id = row.get("subdomain_id", "").strip()
                value = row.get("label", "").strip()

                if not item_id or not value:
                    continue

                label = normalize_subdomain_label(value)

                if label is None:
                    continue

                labels_by_item[item_id][fname] = label

    return labels_by_item, files


# ============================================================
# Build matrix
# ============================================================

def build_matrix(labels_by_item, annotator_files, categories, gold=None):
    """
    Tạo matrix cho Fleiss' Kappa.

    Chỉ giữ item có đủ nhãn từ tất cả annotators.
    """
    matrix = []
    used_item_ids = []
    skipped_missing = 0

    def sort_key(x):
        return int(x) if str(x).isdigit() else str(x)

    for item_id in sorted(labels_by_item.keys(), key=sort_key):
        if gold is not None and item_id not in gold:
            continue

        labels_for_item = labels_by_item[item_id]

        if len(labels_for_item) != len(annotator_files):
            skipped_missing += 1
            continue

        counts = Counter(labels_for_item[f] for f in annotator_files)
        row = [counts.get(cat, 0) for cat in categories]

        if sum(row) != len(annotator_files):
            skipped_missing += 1
            continue

        matrix.append(row)
        used_item_ids.append(item_id)

    return matrix, used_item_ids, skipped_missing


# ============================================================
# Evaluation functions
# ============================================================

def eval_intrusion_kappa(folder):
    print("\n========== FLEISS' KAPPA: TOPIC INTRUSION ==========")

    gold_path = os.path.join(folder, "intrusion_gold.txt")
    gold = read_gold_by_id(gold_path) if os.path.exists(gold_path) else None

    labels_by_item, annotator_files = load_intrusion_labels(folder)

    matrix, used_item_ids, skipped_missing = build_matrix(
        labels_by_item=labels_by_item,
        annotator_files=annotator_files,
        categories=["0", "1", "2", "3", "4", "5"],
        gold=gold
    )

    kappa = fleiss_kappa_from_matrix(matrix)

    print(f"Annotators: {len(annotator_files)}")
    for f in annotator_files:
        print(f"  - {f}")

    print(f"Items used: {len(used_item_ids)}")
    print(f"Items skipped due to missing labels: {skipped_missing}")

    if kappa is None:
        print("Fleiss' Kappa: N/A")
    else:
        print(f"Fleiss' Kappa: {kappa:.4f}")
        print(f"Interpretation: {interpret_kappa(kappa)}")

    return kappa


def eval_subdomain_kappa(folder):
    print("\n========== FLEISS' KAPPA: PARENT-CHILD RELATION ==========")

    gold_path = os.path.join(folder, "subdomain_gold.txt")
    gold = read_gold_by_id(gold_path) if os.path.exists(gold_path) else None

    labels_by_item, annotator_files = load_subdomain_labels(folder)

    matrix, used_item_ids, skipped_missing = build_matrix(
        labels_by_item=labels_by_item,
        annotator_files=annotator_files,
        categories=["y", "n"],
        gold=gold
    )

    kappa = fleiss_kappa_from_matrix(matrix)

    print(f"Annotators: {len(annotator_files)}")
    for f in annotator_files:
        print(f"  - {f}")

    print(f"Items used: {len(used_item_ids)}")
    print(f"Items skipped due to missing labels: {skipped_missing}")

    if kappa is None:
        print("Fleiss' Kappa: N/A")
    else:
        print(f"Fleiss' Kappa: {kappa:.4f}")
        print(f"Interpretation: {interpret_kappa(kappa)}")

    return kappa


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-folder",
        required=True,
        help="Thư mục chứa gold files và các file *_labels.csv"
    )
    parser.add_argument(
        "-task",
        choices=["all", "intrusion", "subdomain"],
        default="all",
        help="Chọn task cần tính Kappa"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise FileNotFoundError(f"Folder not found: {args.folder}")

    if args.task in ["all", "intrusion"]:
        eval_intrusion_kappa(args.folder)

    if args.task in ["all", "subdomain"]:
        eval_subdomain_kappa(args.folder)


if __name__ == "__main__":
    main()
