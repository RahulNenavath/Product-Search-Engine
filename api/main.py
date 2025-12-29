from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

import torch

print("[BOOT] torch:", torch.__version__)
print("[BOOT] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[BOOT] gpu:", torch.cuda.get_device_name(0))


# ----------------------------
# Config
# ---------------------------- 
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USE_SSL = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"
OPENSEARCH_VERIFY_CERTS = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "products_bm25")
OPENSEARCH_TIMEOUT = int(os.getenv("OPENSEARCH_TIMEOUT", "30"))
OPENSEARCH_BM25_INDEX = os.getenv("OPENSEARCH_BM25_INDEX", "products_bm25")
OPENSEARCH_HNSW_INDEX = os.getenv("OPENSEARCH_HNSW_INDEX", "products_hnsw")

FULL_TEXT_FIELD = os.getenv("FULL_TEXT_FIELD", "full_text")

device = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDING_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device  # cpu | cuda | mps
)
VECTOR_FIELD = "embedding"


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        use_ssl=OPENSEARCH_USE_SSL,
        verify_certs=OPENSEARCH_VERIFY_CERTS,
        ssl_show_warn=False,
        timeout=OPENSEARCH_TIMEOUT,
    )


client = get_client()
app = FastAPI(title="Product Search API", version="0.1.0")


# ----------------------------
# Schemas
# ----------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User search query string")
    k: int = Field(10, ge=1, le=100, description="Number of results to return")
    # Optional: if you use this metadata field in your index
    filter_source: Optional[str] = Field(None, description="Optional source filter: ESCI or WANDS")


class SearchHit(BaseModel):
    product_id: Optional[str]
    score: float
    full_text: str
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    index: str
    query: str
    k: int
    hits: List[SearchHit]


# ----------------------------
# Helpers
# ----------------------------
def bm25_query_body(query: str, k: int, filter_source: Optional[str]) -> Dict[str, Any]:
    """
    Uses multi_match if you have common title fields in metadata, otherwise falls back to full_text match.
    Adjust boosts to your needs.
    """
    must_clause: Dict[str, Any] = {
        "multi_match": {
            "query": query,
            "fields": [
                "metadata.product_title^4",
                "metadata.product_name^4",
                "full_text",
            ],
            "type": "best_fields",
        }
    }

    if filter_source:
        return {
            "size": k,
            "query": {
                "bool": {
                    "must": must_clause,
                    "filter": [{"term": {"source": filter_source}}],
                }
            },
        }

    return {"size": k, "query": must_clause}

def hnsw_query_body(
    query_vector: List[float],
    k: int,
    filter_source: Optional[str] = None,
) -> Dict[str, Any]:

    knn_query = {
        "knn": {
            VECTOR_FIELD: {
                "vector": query_vector,
                "k": k,
            }
        }
    }

    if filter_source:
        # Note: filtering is applied on the candidate results returned by ANN in this approach
        return {
            "size": k,
            "query": {
                "bool": {
                    "filter": [{"term": {"source": filter_source}}],
                    "must": [knn_query],
                }
            },
        }

    return {
        "size": k,
        "query": knn_query,
    }


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        info = client.info()
        return {"ok": True, "opensearch": {"cluster_name": info.get("cluster_name"), "version": info.get("version")}}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"OpenSearch not reachable: {e}")


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    try:
        body = bm25_query_body(req.query, req.k, req.filter_source)
        res = client.search(index=OPENSEARCH_INDEX, body=body)
        hits_raw = res.get("hits", {}).get("hits", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    hits: List[SearchHit] = []
    for h in hits_raw:
        src = h.get("_source") or {}
        hits.append(
            SearchHit(
                product_id=src.get("product_id"),
                score=float(h.get("_score", 0.0)),
                source=src.get("source"),
                metadata=src.get("metadata") or {},
                full_text=src.get("full_text") or "",
            )
        )

    return SearchResponse(index=OPENSEARCH_INDEX, query=req.query, k=req.k, hits=hits)


@app.post("/search/bm25", response_model=SearchResponse)
def search_bm25(req: SearchRequest) -> SearchResponse:
    try:
        body = bm25_query_body(req.query, req.k, req.filter_source)
        res = client.search(index=OPENSEARCH_BM25_INDEX, body=body)
        hits_raw = res.get("hits", {}).get("hits", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BM25 search failed: {e}")

    hits: List[SearchHit] = []
    for h in hits_raw:
        src = h.get("_source") or {}
        hits.append(
            SearchHit(
                product_id=src.get("product_id"),
                score=float(h.get("_score", 0.0)),
                source=src.get("source"),
                metadata=src.get("metadata") or {},
                full_text=src.get("full_text") or "",
            )
        )

    return SearchResponse(
        index=OPENSEARCH_BM25_INDEX,
        query=req.query,
        k=req.k,
        hits=hits,
    )

@app.post("/search/hnsw", response_model=SearchResponse)
def search_hnsw(req: SearchRequest) -> SearchResponse:
    try:
        query_vector = EMBEDDING_MODEL.encode(
            req.query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        body = hnsw_query_body(query_vector, req.k, req.filter_source)
        res = client.search(index=OPENSEARCH_HNSW_INDEX, body=body)
        hits_raw = res.get("hits", {}).get("hits", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HNSW search failed: {e}")

    hits: List[SearchHit] = []
    for h in hits_raw:
        src = h.get("_source") or {}
        hits.append(
            SearchHit(
                product_id=src.get("product_id"),
                score=float(h.get("_score", 0.0)),
                source=src.get("source"),
                metadata=src.get("metadata") or {},
                full_text=src.get("full_text") or "",
            )
        )

    return SearchResponse(
        index=OPENSEARCH_HNSW_INDEX,
        query=req.query,
        k=req.k,
        hits=hits,
    )