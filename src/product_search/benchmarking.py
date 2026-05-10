"""
benchmarking.py

Batch benchmark script (no FastAPI) that:
  1) Loads test_query_table (parquet/csv) with columns: [query_id, query]
  2) Loads test_qrels.json (ground truth): {query_id: {doc_id: gain}}
  3) Retrieves ranked results from:
        - BM25 index (match on full_text)
        - HNSW (k-NN on embedding field)
        - Hybrid RRF (BM25 + HNSW fused via Reciprocal Rank Fusion)
        - Hybrid Rerank (BM25 + HNSW candidates + cross-encoder reranking)
  4) Evaluates using SearchEvaluator (Recall/MRR binary optional, NDCG graded)

All embedding and reranking is delegated to the remote embedding service
(EMBEDDING_SERVICE_URL). No ML models are loaded locally.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm

from product_search.evaluation import PredictedRankings, SearchEvaluator
from product_search.search_pipeline import OpenSearchInference


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_query_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


# ── Retrieval functions ───────────────────────────────────────────────────────

def retrieve_bm25(
    inference_client: OpenSearchInference,
    *,
    query_table: pd.DataFrame,
    topn: int,
    filter_source: Optional[str] = None,
) -> PredictedRankings:
    preds: PredictedRankings = {}
    for row in tqdm(query_table.itertuples(index=False), total=len(query_table), desc="BM25 retrieval"):
        qid = str(row.query_id)
        preds[qid] = inference_client.ranked_ids_bm25(
            query=str(row.query),
            k=topn,
            filter_source=filter_source,
        )
    return preds


def retrieve_hnsw(
    inference_client: OpenSearchInference,
    *,
    query_table: pd.DataFrame,
    topn: int,
    filter_source: Optional[str] = None,
    query_vectors: Optional[list] = None,
) -> PredictedRankings:
    preds: PredictedRankings = {}
    qids = query_table["query_id"].astype(str).tolist()
    vecs = query_vectors

    for qid, qvec in tqdm(zip(qids, vecs), total=len(qids), desc="HNSW retrieval"):
        preds[qid] = inference_client.ranked_ids_hnsw(
            query="",
            vector=qvec,
            k=topn,
            filter_source=filter_source,
        )
    return preds


def retrieve_hybrid_rrf(
    inference_client: OpenSearchInference,
    *,
    query_table: pd.DataFrame,
    topn: int,
    filter_source: Optional[str] = None,
    candidate_pool_size: int = 20,
    rrf_k: int = 60,
    query_vectors: Optional[list] = None,
) -> PredictedRankings:
    preds: PredictedRankings = {}
    qids = query_table["query_id"].astype(str).tolist()
    queries = query_table["query"].astype(str).tolist()
    vecs = query_vectors

    for qid, query, qvec in tqdm(zip(qids, queries, vecs), total=len(qids), desc="Hybrid RRF retrieval"):
        preds[qid] = inference_client.ranked_ids_hybrid_rrf(
            query=query,
            vector=qvec,
            k=topn,
            filter_source=filter_source,
            candidate_pool_size=candidate_pool_size,
            rrf_k=rrf_k,
        )
    return preds


def retrieve_hybrid_rerank(
    inference_client: OpenSearchInference,
    *,
    query_table: pd.DataFrame,
    topn: int,
    filter_source: Optional[str] = None,
    candidate_pool_size: int = 20,
    query_vectors: Optional[list] = None,
) -> PredictedRankings:
    preds: PredictedRankings = {}
    qids = query_table["query_id"].astype(str).tolist()
    queries = query_table["query"].astype(str).tolist()
    vecs = query_vectors

    for qid, query, qvec in tqdm(zip(qids, queries, vecs), total=len(qids), desc="Hybrid Rerank retrieval"):
        preds[qid] = inference_client.ranked_ids_hybrid_rerank(
            query=query,
            vector=qvec,
            k=topn,
            filter_source=filter_source,
            candidate_pool_size=candidate_pool_size,
        )
    return preds


def retrieve_agentic(
    inference_client: OpenSearchInference,
    *,
    query_table: pd.DataFrame,
    topn: int,
    filter_source: Optional[str] = None,
    limit_queries: Optional[int] = None,
) -> PredictedRankings:
    from product_search.agent.graph import run_agent  # lazy import

    rows = list(query_table.itertuples(index=False))
    if limit_queries:
        rows = rows[:limit_queries]

    preds: PredictedRankings = {}
    for row in tqdm(rows, total=len(rows), desc="Agentic retrieval"):
        hits, _ = run_agent(
            query=str(row.query),
            k=topn,
            inference=inference_client,
            filter_source=filter_source,
        )
        preds[str(row.query_id)] = [h.product_id for h in hits if h.product_id]
    return preds


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()

    # OpenSearch connection
    ap.add_argument("--opensearch-host", default=os.getenv("OPENSEARCH_HOST", "localhost"))
    ap.add_argument("--opensearch-port", type=int, default=int(os.getenv("OPENSEARCH_PORT", "9200")))
    ap.add_argument("--use-ssl", action="store_true", default=False)
    ap.add_argument("--verify-certs", action="store_true", default=False)
    ap.add_argument("--timeout", type=int, default=int(os.getenv("OPENSEARCH_TIMEOUT", "60")))

    # Index names
    ap.add_argument("--bm25-index", default=os.getenv("OPENSEARCH_BM25_INDEX", "bm25_index"))
    ap.add_argument("--hnsw-index", default=os.getenv("OPENSEARCH_HNSW_INDEX", "hnsw_index"))

    # Embedding service
    ap.add_argument("--embedding-service-url", default=os.getenv("CLOUDRUN_URL") or os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001"))
    ap.add_argument("--embedding-service-timeout", type=int, default=60)
    ap.add_argument("--encode-batch-size", type=int, default=64)

    # Data
    ap.add_argument("--test-qrels", required=True, help="Path to test_qrels.json")
    ap.add_argument("--test-query-table", required=True, help="Path to query table (.parquet or .csv)")

    # Retrieval config
    ap.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--binary-threshold", type=float, default=None)
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--candidate-pool-size", type=int, default=20)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--filter-source", default="", help="Optional: ESCI or WANDS. Empty = no filter.")

    # Run control
    ap.add_argument("--skip-bm25", action="store_true", default=False)
    ap.add_argument("--skip-hnsw", action="store_true", default=False)
    ap.add_argument("--skip-hybrid-rrf", action="store_true", default=False)
    ap.add_argument("--skip-hybrid-rerank", action="store_true", default=False)
    ap.add_argument("--skip-agentic", action="store_true", default=False)
    ap.add_argument("--limit-queries", type=int, default=None,
                    help="Cap number of queries for agentic retrieval (e.g. 50 for quick checks).")
    ap.add_argument("--out-dir", default="./runs")

    args = ap.parse_args()
    filter_source = args.filter_source.strip() or None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_qrels_raw = load_json(Path(args.test_qrels))
    test_queries = load_query_table(args.test_query_table)
    evaluator = SearchEvaluator(ks=args.ks)

    print(f"[INFO] Embedding service: {args.embedding_service_url}")

    # Single shared OpenSearchInference instance for all retrieval methods.
    # No local models — all embedding/reranking delegated to Cloud Run.
    inference = OpenSearchInference(
        bm25_index=args.bm25_index,
        hnsw_index=args.hnsw_index,
        embedding_service_url=args.embedding_service_url,
        embedding_service_timeout=args.embedding_service_timeout,
        opensearch_host=args.opensearch_host,
        opensearch_port=args.opensearch_port,
        opensearch_use_ssl=args.use_ssl,
        opensearch_verify_certs=args.verify_certs,
        opensearch_timeout=args.timeout,
    )

    # Encode all queries once and reuse across HNSW, Hybrid RRF, and Hybrid Rerank.
    # BM25 is lexical-only and never needs vectors.
    needs_vectors = not (args.skip_hnsw and args.skip_hybrid_rrf and args.skip_hybrid_rerank)
    query_vectors = None
    if needs_vectors:
        queries = test_queries["query"].astype(str).tolist()
        print(f"\n[Encode] Batch-encoding {len(queries)} queries via embedding service (once, shared across HNSW / Hybrid)...")
        query_vectors = inference.encode(queries, normalize=True)
        print(f"[Encode] Done — {len(query_vectors)} vectors cached.")

    # ── BM25 ─────────────────────────────────────────────────────────────────
    if not args.skip_bm25:
        print(f"\n[BM25] top-{args.topn} | index={args.bm25_index} | filter={filter_source}")
        bm25_ranked = retrieve_bm25(
            inference,
            query_table=test_queries,
            topn=args.topn,
            filter_source=filter_source,
        )
        save_json(out_dir / "bm25_ranked.json", bm25_ranked)
        scores = evaluator.evaluate_rankings(
            ground_truth_qrels=gt_qrels_raw,
            predicted_rankings=bm25_ranked,
            binary_threshold=args.binary_threshold,
        )
        scores.to_csv(out_dir / "bm25_metrics.csv", index=False)
        print("\n[BM25] Metrics")
        print(scores.to_string(index=False))

    # ── HNSW ─────────────────────────────────────────────────────────────────
    if not args.skip_hnsw:
        print(f"\n[HNSW] top-{args.topn} | index={args.hnsw_index} | filter={filter_source}")
        hnsw_ranked = retrieve_hnsw(
            inference,
            query_table=test_queries,
            topn=args.topn,
            filter_source=filter_source,
            query_vectors=query_vectors,
        )
        save_json(out_dir / "hnsw_ranked.json", hnsw_ranked)
        scores = evaluator.evaluate_rankings(
            ground_truth_qrels=gt_qrels_raw,
            predicted_rankings=hnsw_ranked,
            binary_threshold=args.binary_threshold,
        )
        scores.to_csv(out_dir / "hnsw_metrics.csv", index=False)
        print("\n[HNSW] Metrics")
        print(scores.to_string(index=False))

    # ── Hybrid RRF ───────────────────────────────────────────────────────────
    if not args.skip_hybrid_rrf:
        print(f"\n[Hybrid RRF] top-{args.topn} | candidate_pool={args.candidate_pool_size} | rrf_k={args.rrf_k}")
        hybrid_rrf_ranked = retrieve_hybrid_rrf(
            inference,
            query_table=test_queries,
            topn=args.topn,
            filter_source=filter_source,
            candidate_pool_size=args.candidate_pool_size,
            rrf_k=args.rrf_k,
            query_vectors=query_vectors,
        )
        save_json(out_dir / "hybrid_rrf_ranked.json", hybrid_rrf_ranked)
        scores = evaluator.evaluate_rankings(
            ground_truth_qrels=gt_qrels_raw,
            predicted_rankings=hybrid_rrf_ranked,
            binary_threshold=args.binary_threshold,
        )
        scores.to_csv(out_dir / "hybrid_rrf_metrics.csv", index=False)
        print("\n[Hybrid RRF] Metrics")
        print(scores.to_string(index=False))

    # ── Hybrid Rerank ─────────────────────────────────────────────────────────
    if not args.skip_hybrid_rerank:
        print(f"\n[Hybrid Rerank] top-{args.topn} | candidate_pool={args.candidate_pool_size}")
        hybrid_rerank_ranked = retrieve_hybrid_rerank(
            inference,
            query_table=test_queries,
            topn=args.topn,
            filter_source=filter_source,
            candidate_pool_size=args.candidate_pool_size,
            query_vectors=query_vectors,
        )
        save_json(out_dir / "hybrid_rerank_ranked.json", hybrid_rerank_ranked)
        scores = evaluator.evaluate_rankings(
            ground_truth_qrels=gt_qrels_raw,
            predicted_rankings=hybrid_rerank_ranked,
            binary_threshold=args.binary_threshold,
        )
        scores.to_csv(out_dir / "hybrid_rerank_metrics.csv", index=False)
        print("\n[Hybrid Rerank] Metrics")
        print(scores.to_string(index=False))

    # ── Agentic ───────────────────────────────────────────────────────────────
    if not args.skip_agentic:
        n_label = f"first {args.limit_queries}" if args.limit_queries else "all"
        print(f"\n[Agentic] top-{args.topn} | candidate_pool=25 (initial) | queries={n_label}")
        agentic_ranked = retrieve_agentic(
            inference,
            query_table=test_queries,
            topn=args.topn,
            filter_source=filter_source,
            limit_queries=args.limit_queries,
        )
        save_json(out_dir / "agentic_ranked.json", agentic_ranked)
        scores = evaluator.evaluate_rankings(
            ground_truth_qrels=gt_qrels_raw,
            predicted_rankings=agentic_ranked,
            binary_threshold=args.binary_threshold,
        )
        scores.to_csv(out_dir / "agentic_metrics.csv", index=False)
        print("\n[Agentic] Metrics")
        print(scores.to_string(index=False))

    print(f"\n[DONE] Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
