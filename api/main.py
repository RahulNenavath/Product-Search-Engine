"""
api/main.py

FastAPI product search API backed by OpenSearch.

All search logic is delegated to OpenSearchInference (search_pipeline.py),
which owns BM25, HNSW, and hybrid query construction.
Embedding and reranking are offloaded to the remote embedding service
(running on Cloud Run GPU) — this process loads no ML models.

Endpoints:
  GET  /health                 – liveness + OpenSearch connectivity check
  POST /search/bm25            – BM25 keyword search
  POST /search/hnsw            – Dense-vector ANN search (fine-tuned encoder)
  POST /search/hybrid          – Hybrid BM25 + HNSW via Reciprocal Rank Fusion
  POST /search/hybrid_rerank   – Hybrid BM25 + HNSW + Cross-encoder reranking

Environment variables (all optional, sensible defaults for local dev):
  OPENSEARCH_HOST             default: localhost
  OPENSEARCH_PORT             default: 9200
  OPENSEARCH_USE_SSL          default: false
  OPENSEARCH_VERIFY_CERTS     default: false
  OPENSEARCH_TIMEOUT          default: 30
  OPENSEARCH_BM25_INDEX       default: bm25_index
  OPENSEARCH_HNSW_INDEX       default: hnsw_index
  EMBEDDING_SERVICE_URL       URL of the Cloud Run embedding service
                              default: http://localhost:8001
  EMBEDDING_SERVICE_TIMEOUT   seconds to wait for encode/rerank calls (default: 60)
"""

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from search_pipeline import (
    OpenSearchInference,
    SearchRequest,
    SearchResponse,
)

# Lazy import — avoids pulling in LangChain/VertexAI at startup if unused.
# The agent module is only loaded on the first /search/agentic request.
_run_agent = None

def _get_run_agent():
    global _run_agent
    if _run_agent is None:
        from product_search.agent.graph import run_agent as _ra  # type: ignore
        _run_agent = _ra
    return _run_agent


# ── Config ────────────────────────────────────────────────────────────────────

OPENSEARCH_HOST           = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT           = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USE_SSL        = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"
OPENSEARCH_VERIFY_CERTS   = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"
OPENSEARCH_TIMEOUT        = int(os.getenv("OPENSEARCH_TIMEOUT", "30"))
OPENSEARCH_BM25_INDEX     = os.getenv("OPENSEARCH_BM25_INDEX", "bm25_index")
OPENSEARCH_HNSW_INDEX     = os.getenv("OPENSEARCH_HNSW_INDEX", "hnsw_index")
EMBEDDING_SERVICE_URL     = os.getenv("CLOUDRUN_URL") or os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001")
EMBEDDING_SERVICE_TIMEOUT = int(os.getenv("EMBEDDING_SERVICE_TIMEOUT", "60"))


# ── Boot ──────────────────────────────────────────────────────────────────────

print(f"[BOOT] OpenSearch : {OPENSEARCH_HOST}:{OPENSEARCH_PORT}")
print(f"[BOOT] Embedding service : {EMBEDDING_SERVICE_URL}")

inference = OpenSearchInference(
    bm25_index=OPENSEARCH_BM25_INDEX,
    hnsw_index=OPENSEARCH_HNSW_INDEX,
    embedding_service_url=EMBEDDING_SERVICE_URL,
    embedding_service_timeout=EMBEDDING_SERVICE_TIMEOUT,
    opensearch_host=OPENSEARCH_HOST,
    opensearch_port=OPENSEARCH_PORT,
    opensearch_use_ssl=OPENSEARCH_USE_SSL,
    opensearch_verify_certs=OPENSEARCH_VERIFY_CERTS,
    opensearch_timeout=OPENSEARCH_TIMEOUT,
)

app = FastAPI(title="Product Search API", version="0.1.0")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        info = inference.client.info()
        return {
            "ok": True,
            "opensearch": {
                "cluster_name": info.get("cluster_name"),
                "version": info.get("version"),
            },
            "embedding_service": EMBEDDING_SERVICE_URL,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"OpenSearch not reachable: {e}")


@app.post("/search/bm25", response_model=SearchResponse)
def search_bm25(req: SearchRequest) -> SearchResponse:
    try:
        hits = inference.query_bm25(
            query=req.query,
            k=req.k,
            filter_source=req.filter_source,
            include_full_text=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BM25 search failed: {e}")
    return SearchResponse(index=OPENSEARCH_BM25_INDEX, query=req.query, k=req.k, hits=hits)


@app.post("/search/hnsw", response_model=SearchResponse)
def search_hnsw(req: SearchRequest) -> SearchResponse:
    try:
        hits = inference.query_hnsw(
            query=req.query,
            k=req.k,
            filter_source=req.filter_source,
            include_full_text=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HNSW search failed: {e}")
    return SearchResponse(index=OPENSEARCH_HNSW_INDEX, query=req.query, k=req.k, hits=hits)


@app.post("/search/hybrid", response_model=SearchResponse)
def search_hybrid(req: SearchRequest) -> SearchResponse:
    """Hybrid BM25 + HNSW search fused via Reciprocal Rank Fusion (RRF)."""
    try:
        hits = inference.query_hybrid_rrf(
            query=req.query,
            k=req.k,
            filter_source=req.filter_source,
            candidate_pool_size=50,
            include_full_text=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {e}")
    return SearchResponse(index="hybrid_rrf", query=req.query, k=req.k, hits=hits)


@app.post("/search/agentic", response_model=SearchResponse)
def search_agentic(req: SearchRequest) -> SearchResponse:
    """Agentic search: Gemini Flash rewrites the query, then runs hybrid_rerank.

    The agent optionally reassesses results and retries with a wider candidate
    pool or by decomposing multi-constraint queries (max 2 iterations).
    Latency is higher than /search/hybrid_rerank due to LLM calls.
    """
    try:
        run_agent = _get_run_agent()
        hits, rewritten_query = run_agent(
            query=req.query,
            k=req.k,
            inference=inference,
            filter_source=req.filter_source,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agentic search failed: {e}")
    return SearchResponse(index="agentic", query=req.query, k=req.k, hits=hits, rewritten_query=rewritten_query)


@app.post("/search/hybrid_llm_rerank", response_model=SearchResponse)
def search_hybrid_llm_rerank(req: SearchRequest) -> SearchResponse:
    """Hybrid BM25 + HNSW retrieval re-ranked by Gemini Flash-lite permutation ranking.

    The LLM sees all candidates (pool=25) in one prompt and outputs a ranked
    permutation. Benchmarked at NDCG@5 = 0.730 (+7.0% vs cross-encoder baseline).
    Latency: ~3-5 s per query.
    """
    try:
        hits = inference.query_hybrid_llm_rerank(
            query=req.query,
            k=req.k,
            filter_source=req.filter_source,
            candidate_pool_size=25,
            include_full_text=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid LLM rerank search failed: {e}")
    return SearchResponse(index="hybrid_llm_rerank", query=req.query, k=req.k, hits=hits)


@app.post("/search/hybrid_rerank", response_model=SearchResponse)
def search_hybrid_rerank(req: SearchRequest) -> SearchResponse:
    """Hybrid BM25 + HNSW retrieval followed by cross-encoder reranking.

    Retrieval runs locally against OpenSearch. Reranking is delegated to the
    Cloud Run embedding service (GPU). Latency is dominated by network round
    trip + GPU inference (~1-5 s depending on candidate pool size).
    """
    try:
        hits = inference.query_hybrid_rerank(
            query=req.query,
            k=req.k,
            filter_source=req.filter_source,
            candidate_pool_size=25,
            include_full_text=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid rerank search failed: {e}")
    return SearchResponse(index="hybrid_rerank", query=req.query, k=req.k, hits=hits)
