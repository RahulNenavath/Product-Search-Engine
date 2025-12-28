"""
ingestion.py

Purpose
-------
Build an OpenSearch-ready product corpus from:
  - train_qrels.json  (ground truth; used to select products to ingest)
  - train_query_table.(parquet/csv) (not strictly required for ingest, but validated/loaded for completeness)
  - product_store.json (product_id -> product metadata)

Then:
  1) Build BM25 "full_text" for each product from its metadata
  2) Encode products using a SentenceTransformer to create dense vectors
  3) Create / validate two indices:
       - BM25 index (text)
       - HNSW k-NN index (vector)
  4) Bulk ingest into OpenSearch

Usage
-----
python ingestion.py \
  --opensearch-host localhost \
  --opensearch-port 9200 \
  --train-qrels ./data/train_qrels.json \
  --train-query-table ./data/train_query_table.parquet \
  --product-store ./data/product_store.json \
  --bm25-index products_bm25 \
  --hnsw-index products_hnsw \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 512

Notes
-----
- Assumes OpenSearch 3.x with k-NN plugin for knn_vector/HNSW.
- If you only want BM25 ingestion, pass --no-hnsw.
- Embeddings are computed from "full_text" (recommended). If you truly want only product names,
  use --embed-field product_name (but you’ll likely get worse retrieval).

"""

import argparse
import json
import os
import torch
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from tqdm.auto import tqdm
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer


# ----------------------------
# Model Device Setup
# ----------------------------

def load_embedder(model_name: str) -> tuple[SentenceTransformer, str]:
    """
    Returns (embedder, device_str).
    Prefers Apple Silicon GPU via MPS when available.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    embedder = SentenceTransformer(model_name, device=device)
    return embedder, device


# ----------------------------
# Text construction
# ----------------------------

def build_full_text(meta: Dict[str, Any]) -> str:
    """
    Builds a single BM25 document string from product metadata.
    Works for ESCI + WANDS by checking multiple common keys.
    """
    fields = [
        # ESCI common
        "product_title", "product_brand", "product_color_name",
        "product_bullet_point", "product_description",
        # WANDS/common ecommerce
        "product_name", "brand", "color", "category", "category_hierarchy",
        "product_features", "description", "title",
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


def infer_source_from_id(product_id: str) -> str:
    if product_id.startswith("amz_"):
        return "ESCI"
    if product_id.startswith("wands_"):
        return "WANDS"
    return "UNKNOWN"


# ----------------------------
# Loading artifacts
# ----------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_query_table(path: str) -> Any:
    # Lazy import: avoids pandas dependency if you don't care
    import pandas as pd
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError("train_query_table must be .parquet or .csv")
    if not {"query_id", "query"}.issubset(set(df.columns)):
        raise ValueError("train_query_table_df must contain columns: ['query_id', 'query']")
    return df


def collect_product_ids_from_qrels(qrels: Dict[str, Dict[str, Any]]) -> Set[str]:
    product_ids: Set[str] = set()
    for _, doc_gains in qrels.items():
        if not doc_gains:
            continue
        for pid in doc_gains.keys():
            product_ids.add(str(pid))
    return product_ids


# ----------------------------
# OpenSearch index creation
# ----------------------------

def create_bm25_index(client: OpenSearch, index_name: str) -> None:
    if client.indices.exists(index=index_name):
        return
    body = {
        "settings": {
            "index": {"number_of_shards": 1, "number_of_replicas": 0}
        },
        "mappings": {
            "properties": {
                "product_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "full_text": {"type": "text"},
                "metadata": {"type": "object", "enabled": True},
            }
        },
    }
    client.indices.create(index=index_name, body=body)


def create_hnsw_index(
    client: OpenSearch,
    index_name: str,
    *,
    dim: int,
    vector_field: str = "embedding",
    space_type: str = "cosinesimil",
    engine: str = "faiss",
    m: int = 16,
    ef_construction: int = 128,
    ef_search: int = 100,
) -> None:
    if client.indices.exists(index=index_name):
        return

    body = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "knn.algo_param.ef_search": ef_search,
            }
        },
        "mappings": {
            "properties": {
                "product_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "full_text": {"type": "text"},
                "metadata": {"type": "object", "enabled": True},
                vector_field: {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": space_type,
                        "engine": engine,
                        "parameters": {
                            "ef_construction": ef_construction,
                            "m": m,
                        },
                    },
                },
            }
        },
    }
    client.indices.create(index=index_name, body=body)


# ----------------------------
# Bulk ingestion
# ----------------------------

def batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def bulk_ingest_bm25(
    client: OpenSearch,
    *,
    index_name: str,
    product_store: Dict[str, Dict[str, Any]],
    product_ids: List[str],
    batch_size: int = 1000,
) -> None:
    total = len(product_ids)

    with tqdm(total=total, desc=f"BM25 ingest → {index_name}", unit="docs") as pbar:
        for chunk in batched(product_ids, batch_size):
            actions = []
            for pid in chunk:
                meta = product_store.get(pid)
                if meta is None:
                    continue
                doc = {
                    "product_id": pid,
                    "source": infer_source_from_id(pid),
                    "full_text": build_full_text(meta),
                    "metadata": meta,
                }
                actions.append({
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": pid,
                    "_source": doc,
                })

            if actions:
                helpers.bulk(client, actions)

            pbar.update(len(chunk))

    client.indices.refresh(index=index_name)


def bulk_ingest_hnsw(
    client: OpenSearch,
    *,
    index_name: str,
    product_store: Dict[str, Dict[str, Any]],
    product_ids: List[str],
    embedder: SentenceTransformer,
    dim: int,
    vector_field: str = "embedding",
    batch_size: int = 256,
    encode_batch_size: int = 64,
    normalize_embeddings: bool = True,
) -> None:
    total = len(product_ids)

    with tqdm(total=total, desc=f"HNSW ingest → {index_name}", unit="docs") as pbar:
        for chunk in batched(product_ids, batch_size):
            metas = [product_store.get(pid) for pid in chunk]
            ids_and_texts: List[Tuple[str, str, Dict[str, Any]]] = []

            for pid, meta in zip(chunk, metas):
                if meta is None:
                    continue
                ids_and_texts.append((pid, build_full_text(meta), meta))

            if not ids_and_texts:
                pbar.update(len(chunk))
                continue

            texts = [t for _, t, _ in ids_and_texts]

            vecs = embedder.encode(
                texts,
                batch_size=min(encode_batch_size, len(texts)),
                show_progress_bar=False,  # tqdm already used
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
            )

            vecs_list = vecs.tolist()

            actions = []
            for (pid, text, meta), vec in zip(ids_and_texts, vecs_list):
                if len(vec) != dim:
                    raise ValueError(
                        f"Embedding dimension mismatch for {pid}: got {len(vec)} expected {dim}"
                    )
                doc = {
                    "product_id": pid,
                    "source": infer_source_from_id(pid),
                    "full_text": text,
                    "metadata": meta,
                    vector_field: vec,
                }
                actions.append({
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": pid,
                    "_source": doc,
                })

            if actions:
                helpers.bulk(client, actions)

            pbar.update(len(chunk))

    client.indices.refresh(index=index_name)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opensearch-host", default=os.getenv("OPENSEARCH_HOST", "localhost"))
    ap.add_argument("--opensearch-port", type=int, default=int(os.getenv("OPENSEARCH_PORT", "9200")))
    ap.add_argument("--use-ssl", action="store_true", default=False)
    ap.add_argument("--verify-certs", action="store_true", default=False)
    ap.add_argument("--timeout", type=int, default=int(os.getenv("OPENSEARCH_TIMEOUT", "60")))

    ap.add_argument("--train-qrels", required=True)
    ap.add_argument("--train-query-table", required=True)
    ap.add_argument("--product-store", required=True)

    ap.add_argument("--bm25-index", default=os.getenv("OPENSEARCH_BM25_INDEX", "products_bm25"))
    ap.add_argument("--hnsw-index", default=os.getenv("OPENSEARCH_HNSW_INDEX", "products_hnsw"))
    ap.add_argument("--vector-field", default="embedding")

    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--hnsw-batch-size", type=int, default=256)
    ap.add_argument("--encode-batch-size", type=int, default=64)

    ap.add_argument("--no-hnsw", action="store_true", default=False)
    ap.add_argument("--recreate-indices", action="store_true", default=False)

    args = ap.parse_args()

    # Connect
    client = OpenSearch(
        hosts=[{"host": args.opensearch_host, "port": args.opensearch_port}],
        use_ssl=args.use_ssl,
        verify_certs=args.verify_certs,
        ssl_show_warn=False,
        timeout=args.timeout,
    )

    # Load artifacts
    train_qrels = load_json(args.train_qrels)  # qid -> {pid: gain}
    _ = load_query_table(args.train_query_table)  # validates schema; not required for ingest
    product_store: Dict[str, Dict[str, Any]] = load_json(args.product_store)

    # Select products to ingest (only those appearing in qrels)
    product_ids_set = collect_product_ids_from_qrels(train_qrels)
    product_ids = sorted(product_ids_set)

    missing = [pid for pid in product_ids if pid not in product_store]
    if missing:
        print(f"[WARN] {len(missing)} product_ids from qrels are missing in product_store. "
              f"Example: {missing[:5]}")

    # (Re)create indices
    if args.recreate_indices:
        if client.indices.exists(index=args.bm25_index):
            client.indices.delete(index=args.bm25_index)
        if (not args.no_hnsw) and client.indices.exists(index=args.hnsw_index):
            client.indices.delete(index=args.hnsw_index)

    create_bm25_index(client, args.bm25_index)
    print(f"[OK] BM25 index ready: {args.bm25_index}")

    # Ingest BM25
    t0 = time.time()  # type: ignore[name-defined]
    bulk_ingest_bm25(
        client,
        index_name=args.bm25_index,
        product_store=product_store,
        product_ids=product_ids,
        batch_size=args.batch_size,
    )
    print(f"[OK] BM25 ingested: {args.bm25_index} (docs={client.count(index=args.bm25_index)['count']})")

    # Ingest HNSW
    if not args.no_hnsw:
        embedder, device = load_embedder(args.model)
        print(f"[INFO] SentenceTransformer device = {device}")

        dim = int(embedder.get_sentence_embedding_dimension())

        create_hnsw_index(
            client,
            args.hnsw_index,
            dim=dim,
            vector_field=args.vector_field,
            space_type="cosinesimil",
            engine="nmslib",
        )
        print(f"[OK] HNSW index ready: {args.hnsw_index} (dim={dim})")

        bulk_ingest_hnsw(
            client,
            index_name=args.hnsw_index,
            product_store=product_store,
            product_ids=product_ids,
            embedder=embedder,
            dim=dim,
            vector_field=args.vector_field,
            batch_size=args.hnsw_batch_size,
            encode_batch_size=args.encode_batch_size,
            normalize_embeddings=True,
        )
        print(f"[OK] HNSW ingested: {args.hnsw_index} (docs={client.count(index=args.hnsw_index)['count']})")

    print("[DONE] Ingestion complete.")


if __name__ == "__main__":
    import time
    main()