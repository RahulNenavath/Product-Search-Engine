import os
import requests
from dataclasses import dataclass
from typing import Sequence, Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from opensearchpy import OpenSearch
from collections import defaultdict


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User search query string")
    k: int = Field(10, ge=1, le=100, description="Number of results to return")
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


class EmbeddingServiceError(RuntimeError):
    pass


class OpenSearchInference:
    def __init__(
        self,
        bm25_index: str,
        hnsw_index: str,
        embedding_service_url: str,
        embedding_service_timeout: int = 30,
        opensearch_host: str = os.getenv("OPENSEARCH_HOST", "localhost"),
        opensearch_port: int = int(os.getenv("OPENSEARCH_PORT", "9200")),
        opensearch_use_ssl: bool = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true",
        opensearch_verify_certs: bool = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true",
        opensearch_timeout: int = int(os.getenv("OPENSEARCH_TIMEOUT", "30")),
    ):
        self.bm25_index = bm25_index
        self.hnsw_index = hnsw_index
        self._emb_url = embedding_service_url.rstrip("/")
        self._emb_timeout = embedding_service_timeout

        # Persistent HTTP session for connection pooling to the embedding service
        self._session = requests.Session()

        self.opensearch_config = {
            "host": opensearch_host,
            "port": opensearch_port,
            "use_ssl": opensearch_use_ssl,
            "verify_certs": opensearch_verify_certs,
            "timeout": opensearch_timeout,
        }
        self.client = self.get_client()

    def get_client(self) -> OpenSearch:
        return OpenSearch(
            hosts=[{"host": self.opensearch_config["host"], "port": self.opensearch_config["port"]}],
            use_ssl=self.opensearch_config["use_ssl"],
            verify_certs=self.opensearch_config["verify_certs"],
            ssl_show_warn=False,
            timeout=self.opensearch_config["timeout"],
        )

    # ── Embedding service calls ───────────────────────────────────────────────

    def encode(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """
        Encode a batch of texts to embedding vectors via the remote embedding service.

        Accepts multiple texts at once for efficiency — the service runs GPU-batched
        inference. Used directly by benchmarking for bulk query pre-encoding.
        """
        try:
            resp = self._session.post(
                f"{self._emb_url}/encode",
                json={"texts": texts, "normalize": normalize},
                timeout=self._emb_timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise EmbeddingServiceError(f"Embedding service /encode failed: {e}") from e
        return resp.json()["embeddings"]

    def _rerank_scores(self, query: str, texts: List[str]) -> List[float]:
        """Score (query, text) pairs with the cross-encoder via the remote service."""
        try:
            resp = self._session.post(
                f"{self._emb_url}/rerank",
                json={"query": query, "texts": texts},
                timeout=self._emb_timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise EmbeddingServiceError(f"Embedding service /rerank failed: {e}") from e
        return resp.json()["scores"]

    # ── OpenSearch query builders ─────────────────────────────────────────────

    @staticmethod
    def bm25_query_body(query: str, k: int, filter_source: Optional[str]) -> Dict[str, Any]:
        """
        Matches against full_text, which is the enriched text field written by
        DocumentBuilder.for_bm25() during ingestion. It concatenates title, brand,
        bullets/features, and description — so a single match here covers all signals.
        """
        must_clause: Dict[str, Any] = {
            "match": {
                "full_text": {"query": query},
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

    @staticmethod
    def hnsw_query_body(
        query_vector: List[float],
        k: int,
        filter_source: Optional[str] = None,
    ) -> Dict[str, Any]:

        knn_query = {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k,
                }
            }
        }

        if filter_source:
            return {
                "size": k,
                "query": {
                    "bool": {
                        "filter": [{"term": {"source": filter_source}}],
                        "must": [knn_query],
                    }
                },
            }

        return {"size": k, "query": knn_query}

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: Sequence[Sequence["SearchHit"]],
        *,
        rrf_k: int = 60,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Returns ranked (doc_id, fused_score) pairs.
        fused_score(d) = sum_{lists} 1 / (rrf_k + rank(d))
        """
        scores: Dict[str, float] = defaultdict(float)
        for lst in ranked_lists:
            for rank, hit in enumerate(lst, start=1):
                doc_id = hit.product_id
                if not doc_id:
                    continue
                scores[str(doc_id)] += 1.0 / (rrf_k + rank)

        fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return fused[:top_k]

    # ── Public search methods ─────────────────────────────────────────────────

    def query_bm25(
        self,
        query: str,
        index: Optional[str] = None,
        *,
        k: int = 10,
        filter_source: Optional[str] = None,
        include_full_text: bool = False,
    ) -> List[SearchHit]:

        if index is None:
            index = self.bm25_index

        body = self.bm25_query_body(query, k, filter_source)
        res = self.client.search(index=index, body=body)
        hits_raw = res.get("hits", {}).get("hits", [])

        hits: List[SearchHit] = []
        for h in hits_raw:
            src = h.get("_source") or {}
            hits.append(
                SearchHit(
                    product_id=src.get("product_id") or h.get("_id"),
                    score=float(h.get("_score", 0.0)),
                    source=src.get("source"),
                    metadata=src.get("metadata") or {},
                    full_text=src.get("full_text") if include_full_text else "",
                )
            )
        return hits

    def query_hnsw(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        index: Optional[str] = None,
        *,
        k: int = 10,
        filter_source: Optional[str] = None,
        include_full_text: bool = False,
    ) -> List[SearchHit]:
        """
        If `vector` is provided (e.g. from a pre-computed batch encode), it is used
        directly — no call to the embedding service is made. Otherwise the query
        string is encoded on demand.
        """
        if vector is not None:
            vec = vector
        else:
            vec = self.encode([query])[0]

        body = self.hnsw_query_body(vec, k, filter_source=filter_source)
        res = self.client.search(index=index if index else self.hnsw_index, body=body)
        hits_raw = res.get("hits", {}).get("hits", [])

        hits: List[SearchHit] = []
        for h in hits_raw:
            src = h.get("_source") or {}
            hits.append(
                SearchHit(
                    product_id=src.get("product_id") or h.get("_id"),
                    score=float(h.get("_score", 0.0)),
                    source=src.get("source"),
                    metadata=src.get("metadata") or {},
                    full_text=src.get("full_text") if include_full_text else "",
                )
            )
        return hits

    def query_hybrid_rrf(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        bm25_index: Optional[str] = None,
        hnsw_index: Optional[str] = None,
        *,
        k: int = 10,
        filter_source: Optional[str] = None,
        candidate_pool_size: int = 20,
        include_full_text: bool = False,
        rrf_k: int = 60,
    ) -> List[SearchHit]:
        """
        1) Retrieve top-C from BM25 and HNSW
        2) Fuse via RRF
        3) Return top-k
        """
        bm25_hits = self.query_bm25(
            query=query,
            index=bm25_index if bm25_index else self.bm25_index,
            k=candidate_pool_size,
            filter_source=filter_source,
            include_full_text=include_full_text,
        )

        hnsw_hits = self.query_hnsw(
            query=query,
            vector=vector,
            index=hnsw_index if hnsw_index else self.hnsw_index,
            k=candidate_pool_size,
            filter_source=filter_source,
            include_full_text=include_full_text,
        )

        fused_pairs = self.reciprocal_rank_fusion([bm25_hits, hnsw_hits], rrf_k=rrf_k, top_k=k)
        if not fused_pairs:
            return []

        by_id: Dict[str, SearchHit] = {}
        for h in bm25_hits:
            if h.product_id:
                by_id.setdefault(str(h.product_id), h)
        for h in hnsw_hits:
            if h.product_id:
                by_id[str(h.product_id)] = h

        out: List[SearchHit] = []
        for pid, fused_score in fused_pairs:
            base = by_id.get(pid)
            if base is None:
                out.append(SearchHit(product_id=pid, score=float(fused_score), source=None, metadata={}, full_text=""))
            else:
                out.append(
                    SearchHit(
                        product_id=base.product_id,
                        score=float(fused_score),
                        source=base.source,
                        metadata=base.metadata,
                        full_text=base.full_text if include_full_text else "",
                    )
                )
        return out

    def query_hybrid_rerank(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        bm25_index: Optional[str] = None,
        hnsw_index: Optional[str] = None,
        *,
        k: int = 10,
        filter_source: Optional[str] = None,
        candidate_pool_size: int = 20,
        include_full_text: bool = True,
    ) -> List[SearchHit]:
        """
        1) Retrieve top-C from BM25 and HNSW
        2) Union + de-duplicate by product_id
        3) Cross-encoder rerank via the remote embedding service
        4) Return top-k sorted by reranker score
        """
        bm25_hits = self.query_bm25(
            query=query,
            index=bm25_index if bm25_index else self.bm25_index,
            k=candidate_pool_size,
            filter_source=filter_source,
            include_full_text=include_full_text,
        )

        hnsw_hits = self.query_hnsw(
            query=query,
            vector=vector,
            index=hnsw_index if hnsw_index else self.hnsw_index,
            k=candidate_pool_size,
            filter_source=filter_source,
            include_full_text=include_full_text,
        )

        by_id: Dict[str, SearchHit] = {}
        for h in bm25_hits:
            if h.product_id:
                by_id.setdefault(str(h.product_id), h)
        for h in hnsw_hits:
            if h.product_id:
                by_id[str(h.product_id)] = h

        candidates = list(by_id.values())
        if not candidates:
            return []

        texts = [h.full_text for h in candidates]
        scores = self._rerank_scores(query, texts)

        scored = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)

        out: List[SearchHit] = []
        for hit, s in scored:
            out.append(
                SearchHit(
                    product_id=hit.product_id,
                    score=float(s),
                    source=hit.source,
                    metadata=hit.metadata,
                    full_text=hit.full_text if include_full_text else "",
                )
            )
        return out[:k]

    # ── Ranked-ID helpers (used by benchmarking) ──────────────────────────────

    def ranked_ids_bm25(
        self, query: str, index: Optional[str] = None, *, k: int = 10, filter_source: Optional[str] = None
    ) -> List[str]:
        return [str(h.product_id) for h in self.query_bm25(query, index=index, k=k, filter_source=filter_source)]

    def ranked_ids_hnsw(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        index: Optional[str] = None,
        *,
        k: int = 10,
        filter_source: Optional[str] = None,
    ) -> List[str]:
        return [
            str(h.product_id)
            for h in self.query_hnsw(query, vector=vector, index=index, k=k, filter_source=filter_source)
        ]

    def ranked_ids_hybrid_rrf(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        bm25_index: Optional[str] = None,
        hnsw_index: Optional[str] = None,
        *,
        k: int = 10,
        filter_source: Optional[str] = None,
        candidate_pool_size: int = 20,
        rrf_k: int = 60,
    ) -> List[str]:
        hits = self.query_hybrid_rrf(
            query=query,
            vector=vector,
            k=k,
            filter_source=filter_source,
            candidate_pool_size=candidate_pool_size,
            rrf_k=rrf_k,
            bm25_index=bm25_index,
            hnsw_index=hnsw_index,
            include_full_text=True,
        )
        return [str(h.product_id) for h in hits]

    def ranked_ids_hybrid_rerank(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        bm25_index: Optional[str] = None,
        hnsw_index: Optional[str] = None,
        *,
        k: int = 10,
        filter_source: Optional[str] = None,
        candidate_pool_size: int = 20,
    ) -> List[str]:
        hits = self.query_hybrid_rerank(
            query=query,
            vector=vector,
            k=k,
            filter_source=filter_source,
            candidate_pool_size=candidate_pool_size,
            bm25_index=bm25_index,
            hnsw_index=hnsw_index,
            include_full_text=True,
        )
        return [str(h.product_id) for h in hits]