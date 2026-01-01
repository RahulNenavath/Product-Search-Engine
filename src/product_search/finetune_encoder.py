import json
import os
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

def build_full_text(meta: Dict[str, Any]) -> str:
    """
    Builds a single BM25 document string from product metadata.
    Works for ESCI + WANDS by checking multiple common keys.
    """
    fields = [
        # ESCI common
        "product_title", "product_brand",
        # WANDS/common ecommerce
        "product_name", "product_class",
    ]
    parts: List[str] = []
    for f in fields:
        v = meta.get(f)
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            parts.append(s)
    return " ".join(parts).strip()


def pick_positive_doc(doc_gains: Dict[str, Any]) -> str | None:
    """
    Choose the single best positive per query (highest gain).
    You can change this strategy (sample, keep top-N, etc.).
    """
    if not doc_gains:
        return None
    # gains may be str/int/float
    best_pid, best_gain = None, None
    for pid, g in doc_gains.items():
        try:
            gg = float(g)
        except Exception:
            continue
        if best_gain is None or gg > best_gain:
            best_gain = gg
            best_pid = str(pid)
    return best_pid


def build_training_dataset(
    qrels: Dict[str, Dict[str, Any]],
    qdf: pd.DataFrame,
    product_store: Dict[str, Dict[str, Any]],
    max_pairs: int
    ) -> List[Tuple[str, str, str]]:
    
    qid_to_query: Dict[str, str] = {
        str(r.query_id): str(r.query) for r in qdf.itertuples(index=False)
        }

    # Build (anchor, positive) pairs
    anchors: List[str] = []
    positives: List[str] = []

    n_skipped_no_query = 0
    n_skipped_no_pos = 0
    n_skipped_missing_product = 0
    n_skipped_empty_text = 0

    for qid, doc_gains in tqdm(qrels.items(), desc="Building training pairs"):
        qid = str(qid)
        query = qid_to_query.get(qid)
        if not query:
            n_skipped_no_query += 1
            continue

        pos_pid = pick_positive_doc(doc_gains)
        if not pos_pid:
            n_skipped_no_pos += 1
            continue

        meta = product_store.get(str(pos_pid))
        if meta is None:
            n_skipped_missing_product += 1
            continue

        full_text = build_full_text(meta)
        if not full_text:
            n_skipped_empty_text += 1
            continue

        anchors.append(query)
        positives.append(full_text)

        if max_pairs and len(anchors) >= max_pairs:
            break

    if not anchors:
        raise RuntimeError("No training pairs were created. Check your qrels/query_table/product_store alignment.")

    print(f"[INFO] training_pairs={len(anchors)}")
    print(f"[INFO] skipped: no_query={n_skipped_no_query}, no_pos={n_skipped_no_pos}, "
          f"missing_product={n_skipped_missing_product}, empty_text={n_skipped_empty_text}")

    train_dataset = Dataset.from_dict({"anchor": anchors, "positive": positives})
    return train_dataset

def get_trainer(
    train_dataset: Dataset,
    base_model_name: str,
    guide_model_name: str,
    device: str,
    mini_batch_size: int,
    margin_strategy: str,
    margin: float,
    output_dir: str,
    epochs: int,
    lr: float,
    warmup_ratio: float,
    weight_decay: float,
    train_batch_size: int,
    ) -> SentenceTransformerTrainer:

    # Base model to fine-tune
    model = SentenceTransformer(base_model_name, device=device)

    # Guide/teacher model used by (Cached)GIST to filter likely false negatives in-batch
    # (This is how you “use BGE-M3 as secondary model for in-batch negative mining/filtering”.)
    guide = SentenceTransformer(guide_model_name, device=device)
    # Loss: CachedGISTEmbedLoss
    # Example usage pattern matches the official / community guidance.
    loss = losses.CachedGISTEmbedLoss(
        model=model,
        guide=guide,
        mini_batch_size=mini_batch_size,
        margin_strategy=margin_strategy,
        margin=margin,
    )

    # Training args (Trainer uses tqdm progress bar by default)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        per_device_train_batch_size=train_batch_size,
        fp16=True,                      # good default on modern GPUs; set False if unstable
        bf16=False,                     # set True if your GPU supports bf16 and you prefer it
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],                   # disable wandb etc. by default
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    return trainer


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This script is intended to run on an NVIDIA GPU machine.")
    device = "cuda"
    print("[BOOT] torch:", torch.__version__)
    print("[BOOT] cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))


    qrels = load_json(train_qrels_path)  # {qid: {pid: gain}}
    qdf = load_query_table(train_query_table_path)  # must have query_id, query
    product_store: Dict[str, Dict[str, Any]] = load_json(product_store_path)

    train_dataset = build_training_dataset(qrels, qdf, product_store)
    trainer = get_trainer(
        train_dataset=train_dataset,
        base_model_name=BASE_MODEL,
        guide_model_name=GUIDE_MODEL,
        device=device,
        mini_batch_size=mini_batch_size,
        margin_strategy=margin_strategy,
        margin=margin,
        output_dir=output_dir,
        epochs=epochs,
        lr=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        train_batch_size=train_batch_size,
    )
    trainer.train()
    trainer.save_model(output_dir)

    finetuned = SentenceTransformer(output_dir, device=device)
    test_text = "red puma socks"
    vec = finetuned.encode(test_text, normalize_embeddings=True)
    print("[DEMO] text:", test_text)
    print("[DEMO] embedding_dim:", vec.shape[0])
    print("[DEMO] first_10:", vec[:10].tolist())