import os
import sys
import time
import subprocess

from config.config import (
    ROOT_DIR,

    PAPERS_FILE,
    KEYWORDS_FILE,
    KEYWORD_CNT_FILE,
    INDEX_FILE,

    OUTPUT_DIR,
    EMBEDDING_DIR,
    TREE_DIR,
    TAXONOMY_DIR,

    EMBEDDING_FILE,
    TAXONOMY_TXT,
    TAXONOMY_JSON,

    SBERT_MODEL,
    USE_FINE_TUNE,
    FINETUNED_MODEL_DIR,

    MAX_DEPTH,
    MIN_CLUSTER_SIZE,
    TOP_K,

    HF_HOME,

    DEVICE,
    GPU_NAME,
)

from common.dataset import DataSet


# ======================================================
# ENV CACHE
# ======================================================

os.environ["HF_HOME"] = HF_HOME


# ======================================================
# SCRIPTS
# ======================================================

SBERT_SCRIPT = os.path.join(
    ROOT_DIR,
    "processed_pipeline",
    "sbert_embedding.py",
)

BUILD_TAXONOMY_SCRIPT = os.path.join(
    ROOT_DIR,
    "processed_pipeline",
    "build_taxonomy.py",
)

FINETUNE_SCRIPT = os.path.join(
    ROOT_DIR,
    "processed_pipeline",
    "finetune_sbert.py",
)


# ======================================================
# UTILS
# ======================================================

def ensure_dirs():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    os.makedirs(TREE_DIR, exist_ok=True)
    os.makedirs(TAXONOMY_DIR, exist_ok=True)
    os.makedirs(HF_HOME, exist_ok=True)

    os.makedirs(
        os.path.dirname(KEYWORD_CNT_FILE),
        exist_ok=True,
    )

    os.makedirs(
        os.path.dirname(INDEX_FILE),
        exist_ok=True,
    )


def check_inputs():

    if not os.path.exists(PAPERS_FILE):
        raise FileNotFoundError(
            f"Không thấy file papers.txt:\n{PAPERS_FILE}"
        )

    if not os.path.exists(KEYWORDS_FILE):
        raise FileNotFoundError(
            f"Không thấy file keywords.txt:\n{KEYWORDS_FILE}"
        )


def run_step(name, cmd):

    print("\n" + "=" * 80)
    print(f"[RUNNING] {name}")
    print("=" * 80)

    start = time.time()

    result = subprocess.run(
        cmd,
        cwd=ROOT_DIR,
    )

    if result.returncode != 0:
        print(f"[ERROR] Step failed: {name}")
        sys.exit(result.returncode)

    print(f"[DONE] {name}")
    print(f"[TIME] {time.time() - start:.2f}s")


def build_intermediate_files():

    print("\n" + "=" * 80)
    print("[STEP 1.5] Build intermediate data files")
    print("=" * 80)

    dataset = DataSet(
        document_file=PAPERS_FILE,
        keyword_file=KEYWORDS_FILE,
    )

    if not os.path.exists(KEYWORD_CNT_FILE):

        print("[BUILD] keyword_cnt.txt")
        print(f"[OUTPUT] {KEYWORD_CNT_FILE}")

        dataset.build_keyword_cnt_file(
            KEYWORD_CNT_FILE
        )

    else:
        print(f"[SKIP] keyword_cnt.txt exists:\n{KEYWORD_CNT_FILE}")

    if not os.path.exists(INDEX_FILE):

        print("[BUILD] index.txt")
        print(f"[OUTPUT] {INDEX_FILE}")

        dataset.build_index_file(
            INDEX_FILE
        )

    else:
        print(f"[SKIP] index.txt exists:\n{INDEX_FILE}")


# ======================================================
# MAIN
# ======================================================

def main():

    ensure_dirs()

    check_inputs()

    python = sys.executable

    print("\n" + "=" * 80)
    print("[SYSTEM INFO]")
    print("=" * 80)

    print(f"[DEVICE] {DEVICE}")
    print(f"[GPU] {GPU_NAME}")

    # ======================================================
    # STEP 0: OPTIONAL SBERT FINE-TUNE
    # ======================================================

    if USE_FINE_TUNE:

        if os.path.exists(FINETUNED_MODEL_DIR):

            print(
                f"[SKIP] Fine-tuned model exists:\n"
                f"{FINETUNED_MODEL_DIR}"
            )

        else:

            if not os.path.exists(FINETUNE_SCRIPT):

                raise FileNotFoundError(
                    f"USE_FINE_TUNE=True nhưng thiếu file:\n"
                    f"{FINETUNE_SCRIPT}"
                )

            run_step(
                "0. Fine-tune SBERT",
                [
                    python,
                    FINETUNE_SCRIPT,
                ],
            )

    # ======================================================
    # STEP 1: GLOBAL SBERT EMBEDDING
    # ======================================================

    if not os.path.exists(EMBEDDING_FILE):

        run_step(
            "1. Global SBERT embedding -> phrase_embeddings.txt",
            [
                python,
                SBERT_SCRIPT,
                "--keyword_file",
                KEYWORDS_FILE,
                "--output_file",
                EMBEDDING_FILE,
            ],
        )

    else:

        print(
            f"[SKIP] Embedding exists:\n"
            f"{EMBEDDING_FILE}"
        )

    # ======================================================
    # STEP 1.5: AUTO BUILD DATA FILES
    # ======================================================

    build_intermediate_files()

    # ======================================================
    # STEP 2: BUILD TAXONOMY
    # ======================================================

    run_step(
        "2. Build TaxoGen-style recursive taxonomy",
        [
            python,
            BUILD_TAXONOMY_SCRIPT,

            "--document_file",
            PAPERS_FILE,

            "--keyword_file",
            KEYWORDS_FILE,

            "--embedding_file",
            EMBEDDING_FILE,

            "--output_tree_dir",
            TREE_DIR,

            "--output_taxonomy_txt",
            TAXONOMY_TXT,

            "--output_taxonomy_json",
            TAXONOMY_JSON,

            "--max_depth",
            str(MAX_DEPTH),

            "--min_cluster_size",
            str(MIN_CLUSTER_SIZE),

            "--top_k",
            str(TOP_K),
        ],
    )

    print("\n" + "=" * 80)
    print("[SUCCESS] Improved TaxoGen finished")
    print("=" * 80)

    print(f"SBERT model     : {SBERT_MODEL}")
    print(f"Embedding txt   : {EMBEDDING_FILE}")
    print(f"Keyword cnt txt : {KEYWORD_CNT_FILE}")
    print(f"Index txt       : {INDEX_FILE}")
    print(f"Tree folder     : {TREE_DIR}")
    print(f"Taxonomy txt    : {TAXONOMY_TXT}")
    print(f"Taxonomy json   : {TAXONOMY_JSON}")


if __name__ == "__main__":
    main()