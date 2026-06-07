import os
import csv
import argparse
from collections import defaultdict

YES_VALUES = {"y", "yes", "1", "true", "đúng", "dung", "có", "co"}
NO_VALUES = {"n", "no", "0", "false", "sai", "không", "khong"}


def read_gold_by_id(gold_file):
    gold = {}

    with open(gold_file, "r", encoding="utf-8") as f:

        # CSV của web bắt đầu từ ID = 2
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


def open_csv_dict(path):
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
    return {
        str(k).strip(): "" if v is None else str(v).strip()
        for k, v in row.items()
        if k is not None
    }


def fix_broken_intrusion_row(row):
    """
    Sửa trường hợp Excel bọc nguyên dòng intrusion CSV vào 1 ô.
    """
    if row is None:
        return row

    keys = list(row.keys())

    has_intrusion_id = "intrusion_id" in keys and row.get("intrusion_id")
    has_outlier_id = "outlier_id" in keys and row.get("outlier_id")

    if has_intrusion_id and has_outlier_id:
        return row

    # Trường hợp DictReader chỉ có 1 cột, thường là user_email chứa cả dòng
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
    """
    if row is None:
        return row

    keys = list(row.keys())

    has_subdomain_id = "subdomain_id" in keys and row.get("subdomain_id")
    has_label = "label" in keys and row.get("label")

    if has_subdomain_id and has_label:
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


def normalize_intrusion_pred(pred, gold_value):
    pred = pred.strip()
    gold_value = gold_value.strip()

    if pred == gold_value:
        return gold_value

    if pred.isdigit() and gold_value.isdigit():
        pred_int = int(pred)
        gold_int = int(gold_value)

        if 1 <= pred_int <= 6 and pred_int - 1 == gold_int:
            return gold_value

    return pred


def eval_intrusion(folder):
    gold = read_gold_by_id(os.path.join(folder, "intrusion_gold.txt"))

    files = [
        f for f in os.listdir(folder)
        if f.endswith("_intrusion_labels.csv")
    ]

    grand_total = defaultdict(int)
    grand_correct = defaultdict(int)

    print("\n========== INTRUSION RESULTS ==========")

    for fname in files:
        total = defaultdict(int)
        correct = defaultdict(int)

        path = os.path.join(folder, fname)
        f, reader = open_csv_dict(path)

        with f:
            for raw_row in reader:
                raw_row = fix_broken_intrusion_row(raw_row)
                row = normalize_header(raw_row)

                intrusion_id = row.get("intrusion_id", "").strip()
                pred = row.get("outlier_id", "").strip()

                if not intrusion_id or not pred:
                    continue

                if intrusion_id not in gold:
                    continue

                method = gold[intrusion_id]["method"]
                gold_value = gold[intrusion_id]["gold"]

                pred = normalize_intrusion_pred(pred, gold_value)

                total[method] += 1
                grand_total[method] += 1

                if pred == gold_value:
                    correct[method] += 1
                    grand_correct[method] += 1

        print(f"\nAnnotator file: {fname}")

        if not total:
            print("  Không có nhãn hợp lệ.")

        for method in total:
            acc = correct[method] / total[method]
            print(f"  {method}: {correct[method]}/{total[method]} = {acc:.4f}")

    print("\n---------- AVERAGE INTRUSION ----------")

    if not grand_total:
        print("Không có dữ liệu intrusion hợp lệ.")

    for method in grand_total:
        acc = grand_correct[method] / grand_total[method]
        print(f"{method}: {grand_correct[method]}/{grand_total[method]} = {acc:.4f}")


def eval_subdomain(folder):
    gold = read_gold_by_id(os.path.join(folder, "subdomain_gold.txt"))

    files = [
        f for f in os.listdir(folder)
        if f.endswith("_subdomain_labels.csv")
    ]

    grand_total = defaultdict(int)
    grand_correct = defaultdict(int)
    grand_incorrect = defaultdict(int)

    print("\n========== SUBDOMAIN RESULTS ==========")

    for fname in files:
        total = defaultdict(int)
        correct = defaultdict(int)
        incorrect = defaultdict(int)

        path = os.path.join(folder, fname)
        f, reader = open_csv_dict(path)

        with f:
            for raw_row in reader:
                raw_row = fix_broken_subdomain_row(raw_row)
                row = normalize_header(raw_row)

                subdomain_id = row.get("subdomain_id", "").strip()
                value = row.get("label", "").strip().lower()

                if not subdomain_id or not value:
                    continue

                if value not in YES_VALUES and value not in NO_VALUES:
                    continue

                if subdomain_id not in gold:
                    continue

                method = gold[subdomain_id]["method"]

                total[method] += 1
                grand_total[method] += 1

                if value in YES_VALUES:
                    correct[method] += 1
                    grand_correct[method] += 1
                else:
                    incorrect[method] += 1
                    grand_incorrect[method] += 1

        print(f"\nAnnotator file: {fname}")

        if not total:
            print("  Không có nhãn hợp lệ.")

        for method in total:
            ra = correct[method] / total[method]
            print(
                f"  {method}: "
                f"y={correct[method]}, "
                f"n={incorrect[method]}, "
                f"total={total[method]}, "
                f"RA={ra:.4f}"
            )

    print("\n---------- AVERAGE SUBDOMAIN ----------")

    if not grand_total:
        print("Không có dữ liệu subdomain hợp lệ.")

    for method in grand_total:
        ra = grand_correct[method] / grand_total[method]
        print(
            f"{method}: "
            f"y={grand_correct[method]}, "
            f"n={grand_incorrect[method]}, "
            f"total={grand_total[method]}, "
            f"RA={ra:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-folder", required=True)
    args = parser.parse_args()

    eval_intrusion(args.folder)
    eval_subdomain(args.folder)