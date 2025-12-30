"""
batch_benchmark.py

Batch benchmark script (no FastAPI) that:
  1) Loads test_query_table (parquet/csv) with columns: [query_id, query]
  2) Loads test_qrels.json (ground truth): {query_id: {doc_id: gain}}
  3) Retrieves ranked results from:
        - BM25 index (match on full_text)
        - HNSW (k-NN on embedding field)
  4) Evaluates using SearchEvaluator (Recall/MRR binary optional, NDCG graded)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm.auto import tqdm
import pandas as pd
import torch
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from product_search.inference import OpenSearchInference
from product_search.evaluator import SearchEvaluator, PredictedRankings

VECTOR_FIELD = "embedding"


# ----------------------------
# Model / device
# ----------------------------
def load_embedder(model_name: str) -> Tuple[SentenceTransformer, str]:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"
    emb = SentenceTransformer(model_name, device=device)
    print(f"[INFO] Embedding model loaded on device={device}")
    return emb, device


# ----------------------------
# Retrieval
# ----------------------------
def retrieve_bm25(
    inference_client: OpenSearchInference,
    *,
    index: str,
    query_table: pd.DataFrame,
    topn: int,
    filter_source: Optional[str] = None,
) -> PredictedRankings:
    preds: PredictedRankings = {}
    
    for row in tqdm(
        query_table.itertuples(index=False), 
        total=len(query_table), 
        desc="BM25 retrieval"
        ):

        qid = str(row.query_id)
        hits = inference_client.ranked_ids_bm25(
            query=str(row.query),
            index=index,
            k=topn,
            filter_source=filter_source,
        )
        preds[qid] = hits
    return preds


def retrieve_hnsw(
    inference_client: OpenSearchInference,
    *,
    index: str,
    query_table: pd.DataFrame,
    embedder: SentenceTransformer,
    topn: int,
    filter_source: Optional[str] = None,
    encode_batch_size: int = 64,
) -> PredictedRankings:
    preds: PredictedRankings = {}

    # Encode in batches for speed
    qids = query_table["query_id"].astype(str).tolist()
    queries = query_table["query"].astype(str).tolist()

    # SentenceTransformer returns np.ndarray; keep as python lists for JSON
    vecs = embedder.encode(
        queries,
        batch_size=min(encode_batch_size, len(queries)),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    for qid, qvec in tqdm(
        zip(qids, vecs),
        total=len(qids),
        desc="HNSW retrieval"
        ):

        hits = inference_client.ranked_ids_hnsw_with_vector(
            vector=qvec.tolist(),
            index=index,
            k=topn,
            filter_source=filter_source,
        )
        preds[qid] = hits

    return preds


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

    ap.add_argument("--bm25-index", default=os.getenv("OPENSEARCH_BM25_INDEX", "products_bm25"))
    ap.add_argument("--hnsw-index", default=os.getenv("OPENSEARCH_HNSW_INDEX", "products_hnsw"))
    ap.add_argument("--vector-field", default=os.getenv("OPENSEARCH_VECTOR_FIELD", "embedding"))

    ap.add_argument("--test-qrels", required=True)
    ap.add_argument("--test-query-table", required=True)

    ap.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    ap.add_argument("--encode-batch-size", type=int, default=64)

    ap.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--binary-threshold", type=float, default=None)
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--filter-source", default="", help="Optional: ESCI or WANDS. Empty means no filter.")
    ap.add_argument("--out-dir", default="./runs")

    ap.add_argument("--skip-hnsw", action="store_true", default=False)
    ap.add_argument("--skip-bm25", action="store_true", default=False)

    args = ap.parse_args()
    filter_source = args.filter_source.strip() or None

    client = OpenSearch(
        hosts=[{"host": args.opensearch_host, "port": args.opensearch_port}],
        use_ssl=args.use_ssl,
        verify_certs=args.verify_certs,
        ssl_show_warn=False,
        timeout=args.timeout,
    )

    gt_qrels_raw = load_json(args.test_qrels)
    test_queries = load_query_table(args.test_query_table)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = SearchEvaluator(ks=args.ks)

    # ---------------- BM25 ----------------
    if not args.skip_bm25:
        print(f"[BM25] Retrieving top-{args.topn} from index={args.bm25_index} filter_source={filter_source}")
        bm25_ranked = retrieve_bm25(
            client,
            index=args.bm25_index,
            query_table=test_queries,
            topn=args.topn,
            filter_source=filter_source,
        )
        save_json(out_dir / "bm25_ranked.json", bm25_ranked)

        bm25_scores = evaluator.evaluate_rankings(
            ground_truth_qrels=gt_qrels_raw,
            predicted_rankings=bm25_ranked,
            binary_threshold=args.binary_threshold,
        )
        bm25_scores.to_csv(out_dir / "bm25_metrics.csv", index=False)
        print("\n[B M 2 5] Metrics")
        print(bm25_scores)

    # ---------------- HNSW ----------------
    if not args.skip_hnsw:
        embedder, device = load_embedder(args.model)
        print(f"[HNSW] Embedder device={device} model={args.model}")
        print(f"[HNSW] Retrieving top-{args.topn} from index={args.hnsw_index} vector_field={args.vector_field} filter_source={filter_source}")

        hnsw_ranked = retrieve_hnsw(
            client,
            index=args.hnsw_index,
            query_table=test_queries,
            embedder=embedder,
            vector_field=args.vector_field,
            topn=args.topn,
            filter_source=filter_source,
            encode_batch_size=args.encode_batch_size,
        )
        save_json(out_dir / "hnsw_ranked.json", hnsw_ranked)

        hnsw_scores = evaluator.evaluate_rankings(
            ground_truth_qrels=gt_qrels_raw,
            predicted_rankings=hnsw_ranked,
            binary_threshold=args.binary_threshold,
        )
        hnsw_scores.to_csv(out_dir / "hnsw_metrics.csv", index=False)
        print("\n[H N S W] Metrics")
        print(hnsw_scores)

    print(f"\n[DONE] Saved outputs to: {out_dir.resolve()}")