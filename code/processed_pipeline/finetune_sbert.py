import os
import sys
import random

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from torch.utils.data import DataLoader

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
)

from config.config import (
    PAPERS_FILE,

    SBERT_MODEL,
    FINETUNED_MODEL_DIR,

    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WARMUP_RATIO,

    MAX_TRAIN_PAIRS,
    PAIR_WINDOW_SIZE,

    DEVICE,
)

from common.utils import read_lines


# ======================================================
# GENERATE TRAIN PAIRS
# ======================================================

def generate_pairs():

    print("[PAIR] Loading documents...")

    documents = [
        line.split()
        for line in read_lines(PAPERS_FILE)
    ]

    pairs = []

    print("[PAIR] Generating positive pairs...")

    for doc in documents:

        tokens = list(set(doc))

        if len(tokens) < 2:
            continue

        random.shuffle(tokens)

        max_window = min(
            PAIR_WINDOW_SIZE,
            len(tokens),
        )

        for i in range(max_window):

            for j in range(i + 1, max_window):

                a = tokens[i]
                b = tokens[j]

                pairs.append((a, b))

                if len(pairs) >= MAX_TRAIN_PAIRS:

                    print(
                        f"[PAIR] Reached {MAX_TRAIN_PAIRS}"
                    )

                    return pairs

    return pairs


# ======================================================
# TRAIN GLOBAL SBERT
# ======================================================

def train_sbert():

    print(f"[DEVICE] {DEVICE}")

    model = SentenceTransformer(
        SBERT_MODEL,
        device=DEVICE,
    )

    pairs = generate_pairs()

    print(f"[PAIR] Total pairs: {len(pairs)}")

    train_examples = [
        InputExample(
            texts=[
                a.replace("_", " "),
                b.replace("_", " "),
            ]
        )
        for a, b in pairs
    ]

    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE,
    )

    train_loss = losses.MultipleNegativesRankingLoss(
        model
    )

    warmup_steps = int(
        len(train_loader)
        * EPOCHS
        * WARMUP_RATIO
    )

    print("[TRAIN] Start fine-tuning...")

    model.fit(
        train_objectives=[
            (train_loader, train_loss)
        ],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={
            "lr": LEARNING_RATE
        },
        show_progress_bar=True,
        output_path=FINETUNED_MODEL_DIR,
        checkpoint_path=None,
    )

    os.makedirs(
        FINETUNED_MODEL_DIR,
        exist_ok=True,
    )

    model.save(FINETUNED_MODEL_DIR)

    print("[DONE] Fine-tuned model saved:")
    print(FINETUNED_MODEL_DIR)


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    train_sbert()