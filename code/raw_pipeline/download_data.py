import os
import sys
import zipfile
import shutil
import gdown


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


from config.config import (
    DATASET_URL,
    DATASET_ZIP_NAME,
)


RAW_DIR = os.path.join(
    ROOT_DIR,
    "data",
    "raw",
)

PROCESSED_DIR = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
)


# ======================================================
# DOWNLOAD
# ======================================================

def download_dataset():

    os.makedirs(RAW_DIR, exist_ok=True)

    zip_path = os.path.join(
        RAW_DIR,
        DATASET_ZIP_NAME,
    )

    if os.path.exists(zip_path):

        print("[SKIP] Dataset already exists")
        print(zip_path)

        return zip_path

    print("[DOWNLOAD]")
    print(DATASET_URL)

    gdown.download(
        DATASET_URL,
        zip_path,
        quiet=False,
    )

    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Download failed: {zip_path}"
        )

    print("[DONE]")
    print(zip_path)

    return zip_path


# ======================================================
# EXTRACT
# ======================================================

def extract_dataset(zip_path):

    extract_dir = os.path.join(
        RAW_DIR,
        "extracted",
    )

    if os.path.exists(extract_dir):

        print("[SKIP] Already extracted")
        print(extract_dir)

        return extract_dir

    os.makedirs(extract_dir, exist_ok=True)

    print("[EXTRACT]")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    print("[DONE]")
    print(extract_dir)

    return extract_dir


# ======================================================
# FIND FILE RECURSIVELY
# ======================================================

def find_file(root, filename):

    for cur_root, dirs, files in os.walk(root):

        if filename in files:

            return os.path.join(
                cur_root,
                filename,
            )

    return None


# ======================================================
# MOVE PROCESSED DATA
# ======================================================

def move_processed_data(extract_dir):

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    files = [
        "papers.txt",
        "keywords.txt",
        "index.txt",
        "keyword_cnt.txt",
    ]

    for file_name in files:

        src = find_file(
            extract_dir,
            file_name,
        )

        if src is None:

            raise FileNotFoundError(
                f"Missing file in zip: {file_name}"
            )

        dst = os.path.join(
            PROCESSED_DIR,
            file_name,
        )

        shutil.copy2(src, dst)

        print(f"[COPY] {file_name}")
        print(f"       {dst}")


# ======================================================
# MAIN
# ======================================================

def main():

    zip_path = download_dataset()

    extract_dir = extract_dataset(zip_path)

    move_processed_data(extract_dir)

    print("\n[SUCCESS] Dataset prepared.")
    print(PROCESSED_DIR)


if __name__ == "__main__":
    main()