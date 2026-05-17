import os
import sys
import argparse
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from sentence_transformers import SentenceTransformer

from common.utils import read_lines

from config.config import (
    SBERT_MODEL,
    FINETUNED_MODEL_DIR,
    USE_FINE_TUNE,
    DEVICE,
    BATCH_SIZE,
    NORMALIZE_EMBEDDINGS,
)

# ======================================================
# SAVE TXT EMBEDDINGS
# ======================================================

def save_txt_embeddings(
    keywords,
    embeddings,
    output_file,
):
    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True,
    )

    with open(output_file, "w", encoding="utf-8") as f:

        dim = embeddings.shape[1]

        # Word2Vec-style header
        f.write(f"{len(keywords)} {dim}\n")

        for kw, vec in zip(keywords, embeddings):

            vec_str = " ".join(
                [str(float(x)) for x in vec]
            )

            f.write(f"{kw} {vec_str}\n")


# ======================================================
# LOAD SBERT MODEL
# ======================================================

def load_sbert_model():
    """
    Nếu đã fine-tune:
        load fine-tuned model.

    Nếu chưa:
        load pretrained SBERT.
    """

    if (
        USE_FINE_TUNE
        and
        os.path.exists(FINETUNED_MODEL_DIR)
    ):
        print(
            "[SBERT] Loading fine-tuned model:"
        )
        print(FINETUNED_MODEL_DIR)

        model_name = FINETUNED_MODEL_DIR

    else:
        print(
            "[SBERT] Loading pretrained model:"
        )
        print(SBERT_MODEL)

        model_name = SBERT_MODEL

    model = SentenceTransformer(
        model_name,
        device=DEVICE,
    )

    return model


# ======================================================
# BUILD GLOBAL EMBEDDINGS
# ======================================================

def build_sbert_embeddings(
    keyword_file,
    output_file,
):
    model = load_sbert_model()

    keywords = read_lines(keyword_file)

    print(
        f"[SBERT] Encoding {len(keywords)} keywords..."
    )

    embeddings = model.encode(
        keywords,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True,
    )

    embeddings = np.array(
        embeddings,
        dtype=np.float32,
    )

    save_txt_embeddings(
        keywords=keywords,
        embeddings=embeddings,
        output_file=output_file,
    )

    print(f"[SBERT] Saved embeddings: {output_file}")
    print(f"[SBERT] Shape: {embeddings.shape}")


# ======================================================
# CLI
# ======================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--keyword_file",
        required=True,
    )

    parser.add_argument(
        "--output_file",
        required=True,
    )

    args = parser.parse_args()

    build_sbert_embeddings(
        keyword_file=args.keyword_file,
        output_file=args.output_file,
    )