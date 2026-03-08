"""
api/main.py

FastAPI product search API backed by OpenSearch.

All search logic is delegated to OpenSearchInference (search_pipeline.py),
which owns BM25, HNSW, and hybrid query construction.

Endpoints:
  GET  /health          – liveness + OpenSearch connectivity check
  POST /search/bm25     – BM25 keyword search
  POST /search/hnsw     – Dense-vector ANN search (fine-tuned encoder)
  POST /search/hybrid   – Hybrid BM25 + HNSW via Reciprocal Rank Fusion

Environment variables (all optional, sensible defaults for local dev):
  OPENSEARCH_HOST          default: localhost
  OPENSEARCH_PORT          default: 9200
  OPENSEARCH_USE_SSL       default: false
  OPENSEARCH_VERIFY_CERTS  default: false
  OPENSEARCH_TIMEOUT       default: 30
  OPENSEARCH_BM25_INDEX    default: bm25_index
  OPENSEARCH_HNSW_INDEX    default: hnsw_index
  EMBEDDING_MODEL_PATH     default: finetuned_encoder
                           (path to the fine-tuned SentenceTransformer directory)
"""

import os
from typing import Any, Dict

import torch
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer

from search_pipeline import (
    OpenSearchInference,
    SearchRequest,
    SearchResponse,
)


# ── Config ────────────────────────────────────────────────────────────────────

OPENSEARCH_HOST         = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT         = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USE_SSL      = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"
OPENSEARCH_VERIFY_CERTS = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"
OPENSEARCH_TIMEOUT      = int(os.getenv("OPENSEARCH_TIMEOUT", "30"))
OPENSEARCH_BM25_INDEX   = os.getenv("OPENSEARCH_BM25_INDEX", "bm25_index")
OPENSEARCH_HNSW_INDEX   = os.getenv("OPENSEARCH_HNSW_INDEX", "hnsw_index")
EMBEDDING_MODEL_PATH    = os.getenv("EMBEDDING_MODEL_PATH", "finetuned_encoder")


# ── Boot ──────────────────────────────────────────────────────────────────────

print("[BOOT] torch:", torch.__version__)
print("[BOOT] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[BOOT] gpu:", torch.cuda.get_device_name(0))


def _load_embedder(model_path: str) -> SentenceTransformer:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[BOOT] Loading embedder from '{model_path}' on {device}")
    return SentenceTransformer(model_path, device=device)


embedder = _load_embedder(EMBEDDING_MODEL_PATH)

inference = OpenSearchInference(
    bm25_index=OPENSEARCH_BM25_INDEX,
    hnsw_index=OPENSEARCH_HNSW_INDEX,
    embedding_model=embedder,
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
            include_full_text=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {e}")
    return SearchResponse(index="hybrid_rrf", query=req.query, k=req.k, hits=hits)
