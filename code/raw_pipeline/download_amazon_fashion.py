import os
import sys
import gzip
import shutil
import requests
from tqdm import tqdm


# ======================================================
# FIX ROOT IMPORT
# ======================================================

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(ROOT_DIR)


# ======================================================
# AMAZON FASHION 2023 URLS
# ======================================================

AMAZON_FASHION_REVIEW_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/"
    "amazon_2023/raw/review_categories/Amazon_Fashion.jsonl.gz"
)

AMAZON_FASHION_META_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/"
    "amazon_2023/raw/meta_categories/meta_Amazon_Fashion.jsonl.gz"
)


RAW_DIR = os.path.join(
    ROOT_DIR,
    "data",
    "raw",
    "amazon_fashion",
)

PROCESSED_DIR = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "amazon_fashion",
)


# ======================================================
# DOWNLOAD WITH PROGRESS BAR
# ======================================================

def download_file(url, output_path):

    if os.path.exists(output_path):

        print("[SKIP] File already exists")
        print(output_path)

        return output_path

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    print("[DOWNLOAD]")
    print(url)

    response = requests.get(
        url,
        stream=True,
        timeout=60,
    )

    response.raise_for_status()

    total_size = int(
        response.headers.get("content-length", 0)
    )

    with open(output_path, "wb") as f:

        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=os.path.basename(output_path),
        ) as pbar:

            for chunk in response.iter_content(chunk_size=1024 * 1024):

                if chunk:

                    f.write(chunk)
                    pbar.update(len(chunk))

    print("[DONE]")
    print(output_path)

    return output_path


# ======================================================
# EXTRACT GZ TO JSONL
# ======================================================

def extract_gz(gz_path, output_path):

    if os.path.exists(output_path):

        print("[SKIP] Already extracted")
        print(output_path)

        return output_path

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    print("[EXTRACT]")
    print(gz_path)

    with gzip.open(gz_path, "rb") as f_in:

        with open(output_path, "wb") as f_out:

            shutil.copyfileobj(f_in, f_out)

    print("[DONE]")
    print(output_path)

    return output_path


# ======================================================
# MAIN
# ======================================================

def main():

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    review_gz = os.path.join(
        RAW_DIR,
        "Amazon_Fashion.jsonl.gz",
    )

    meta_gz = os.path.join(
        RAW_DIR,
        "meta_Amazon_Fashion.jsonl.gz",
    )

    review_jsonl = os.path.join(
        PROCESSED_DIR,
        "Amazon_Fashion.jsonl",
    )

    meta_jsonl = os.path.join(
        PROCESSED_DIR,
        "meta_Amazon_Fashion.jsonl",
    )

    download_file(
        AMAZON_FASHION_REVIEW_URL,
        review_gz,
    )

    download_file(
        AMAZON_FASHION_META_URL,
        meta_gz,
    )

    extract_gz(
        review_gz,
        review_jsonl,
    )

    extract_gz(
        meta_gz,
        meta_jsonl,
    )

    print("\n[SUCCESS] Amazon Fashion 2023 dataset prepared.")
    print(PROCESSED_DIR)


if __name__ == "__main__":
    main()