import random
import numpy as np
from torch.utils.data import DataLoader

from sentence_transformers import InputExample, losses

from config.config import (
    LOCAL_CONTEXT_DOCS,
    LAMBDA_KEYWORD,
    DEVICE,
    USE_LOCAL_FINE_TUNE,
    LOCAL_FINE_TUNE_EPOCHS,
    LOCAL_FINE_TUNE_BATCH_SIZE,
    LOCAL_MAX_TRAIN_PAIRS,
    LOCAL_MIN_TRAIN_PAIRS,
)


def build_node_context(doc_ids, documents, max_docs=LOCAL_CONTEXT_DOCS):
    selected = []

    for doc_id in doc_ids[:max_docs]:
        if doc_id < 0 or doc_id >= len(documents):
            continue

        doc = documents[doc_id]

        if isinstance(doc, list):
            selected.append(" ".join(doc))
        else:
            selected.append(str(doc))

    return " ".join(selected)


def generate_local_positive_pairs(
    doc_ids,
    documents,
    seed_keywords,
    max_pairs=LOCAL_MAX_TRAIN_PAIRS,
):
    keyword_set = set(seed_keywords)
    pairs = []

    for doc_id in doc_ids:
        if doc_id < 0 or doc_id >= len(documents):
            continue

        doc = documents[doc_id]

        kws = [
            token
            for token in doc
            if token in keyword_set
        ]

        kws = list(set(kws))

        if len(kws) < 2:
            continue

        random.shuffle(kws)

        for i in range(len(kws)):
            for j in range(i + 1, len(kws)):
                pairs.append((kws[i], kws[j]))

                if len(pairs) >= max_pairs:
                    return pairs

    return pairs


def local_sbert_finetune(
    sbert_model,
    doc_ids,
    documents,
    seed_keywords,
):
    """
    Fine-tune SBERT thật ở node hiện tại.

    Positive pair:
        hai keyword cùng xuất hiện trong document của node.
    """

    pairs = generate_local_positive_pairs(
        doc_ids=doc_ids,
        documents=documents,
        seed_keywords=seed_keywords,
    )

    if len(pairs) < LOCAL_MIN_TRAIN_PAIRS:
        print(
            f"[LOCAL FINE-TUNE] Skip: only {len(pairs)} pairs"
        )
        return sbert_model

    print(
        f"[LOCAL FINE-TUNE] Training with {len(pairs)} pairs, "
        f"epochs={LOCAL_FINE_TUNE_EPOCHS}"
    )

    train_examples = [
        InputExample(texts=[a.replace("_", " "), b.replace("_", " ")])
        for a, b in pairs
    ]

    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=LOCAL_FINE_TUNE_BATCH_SIZE,
    )

    train_loss = losses.MultipleNegativesRankingLoss(
        sbert_model
    )

    warmup_steps = int(
        len(train_loader) * LOCAL_FINE_TUNE_EPOCHS * 0.1
    )

    sbert_model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=LOCAL_FINE_TUNE_EPOCHS,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    return sbert_model


def build_local_sbert_embeddings(
    seed_keywords,
    keyword_to_id,
    global_embeddings,
    sbert_model,
    context_text,
    lambda_keyword=LAMBDA_KEYWORD,
):
    """
    Local SBERT Re-Embedding.

    local_vec =
        lambda * global_keyword_vec
        + (1-lambda) * context_vec
    """

    valid_keywords = [
        kw
        for kw in seed_keywords
        if kw in keyword_to_id
    ]

    if len(valid_keywords) == 0:
        return [], np.array([])

    keyword_vectors = np.array([
        global_embeddings[keyword_to_id[kw]]
        for kw in valid_keywords
    ])

    if context_text.strip():
        context_vector = sbert_model.encode(
            context_text,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=DEVICE,
        )
    else:
        context_vector = np.mean(keyword_vectors, axis=0)

    local_vectors = []

    for vec in keyword_vectors:
        local_vec = (
            lambda_keyword * vec
            + (1.0 - lambda_keyword) * context_vector
        )

        norm = np.linalg.norm(local_vec)

        if norm > 0:
            local_vec = local_vec / norm

        local_vectors.append(local_vec)

    return valid_keywords, np.array(local_vectors)


def build_local_embeddings_for_node(
    seed_keywords,
    keyword_to_id,
    global_embeddings,
    sbert_model,
    doc_ids,
    documents,
):
    """
    Hàm chính dùng trong build_taxonomy.py.

    Nếu USE_LOCAL_FINE_TUNE=True:
        fine-tune SBERT ở node.
    Sau đó:
        tạo local embedding.
    """

    if USE_LOCAL_FINE_TUNE:
        sbert_model = local_sbert_finetune(
            sbert_model=sbert_model,
            doc_ids=doc_ids,
            documents=documents,
            seed_keywords=seed_keywords,
        )

    context_text = build_node_context(
        doc_ids=doc_ids,
        documents=documents,
        max_docs=LOCAL_CONTEXT_DOCS,
    )

    return build_local_sbert_embeddings(
        seed_keywords=seed_keywords,
        keyword_to_id=keyword_to_id,
        global_embeddings=global_embeddings,
        sbert_model=sbert_model,
        context_text=context_text,
        lambda_keyword=LAMBDA_KEYWORD,
    )