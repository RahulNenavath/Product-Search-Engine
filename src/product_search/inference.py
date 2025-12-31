import os
import torch
from dataclasses import dataclass
from typing import Sequence, Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field
from opensearchpy import OpenSearch
from collections import defaultdict
from typing import Iterable, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder

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


@dataclass
class OpenSearchInference:
    def __init__(self, 
        bm25_index: str, 
        hnsw_index: str,
        embedding_model: SentenceTransformer,
        opensearch_host: str = os.getenv("OPENSEARCH_HOST", "localhost"),
        opensearch_port: int = int(os.getenv("OPENSEARCH_PORT", "9200")),
        opensearch_use_ssl: bool = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true",
        opensearch_verify_certs: bool = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true",
        opensearch_timeout: int = int(os.getenv("OPENSEARCH_TIMEOUT", "30")),
    ):
        self.client = None
        self.bm25_index = bm25_index
        self.hnsw_index = hnsw_index
        self.embedding_model = embedding_model

        self.opensearch_config = {
            "host": opensearch_host,
            "port": opensearch_port,
            "use_ssl": opensearch_use_ssl,
            "verify_certs": opensearch_verify_certs,
            "timeout": opensearch_timeout,
        }

        if self.client is None:
            self.client = self.get_client()
    
    def get_client(self) -> OpenSearch:
        return OpenSearch(
            hosts=[{"host": self.opensearch_config["host"], "port": self.opensearch_config["port"]}],
            use_ssl=self.opensearch_config["use_ssl"],
            verify_certs=self.opensearch_config["verify_certs"],
            ssl_show_warn=False,
            timeout=self.opensearch_config["timeout"]
            )
    
    @staticmethod
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
                "k": k
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

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: Sequence[Sequence[SearchHit]],
        *,
        rrf_k: int = 60,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Returns ranked (doc_id, fused_score) pairs.
        fused_score(d) = sum_{lists} 1 / (rrf_k + rank(d))
        """
        scores = defaultdict(float)

        for lst in ranked_lists:
            for rank, hit in enumerate(lst, start=1):
                doc_id = hit.product_id
                if not doc_id:
                    continue
                scores[str(doc_id)] += 1.0 / (rrf_k + rank)

        fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return fused[:top_k]

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
                    full_text=src.get("full_text") if include_full_text else ""
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
        normalize_embeddings: bool = True,
        include_full_text: bool = False,
    ) -> List[SearchHit]:

        if vector is not None:
            vec = vector
        else:
            vec = self.embedding_model.encode(
                query,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
            ).tolist()
        
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
                    full_text=src.get("full_text") if include_full_text else ""
                    )
                )
        return hits

    def query_hybrid_rrf(
        self,
        query: str,
        vector : Optional[List[float]] = None,
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
        3) Return top-k doc_ids
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

        fused_pairs = self.reciprocal_rank_fusion(
            [bm25_hits, hnsw_hits], 
            rrf_k=rrf_k, 
            top_k=k
            )
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
                out.append(
                    SearchHit(
                        product_id=pid, 
                        score=float(fused_score), 
                        source=None, 
                        metadata={}, 
                        full_text=""
                    ))
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

    def ranked_ids_bm25(self, query: str, index: Optional[str] = None, *, k: int = 10, filter_source: Optional[str] = None) -> List[str]:
        return [str(h.product_id) for h in self.query_bm25(query, index=index, k=k, filter_source=filter_source)]

    def ranked_ids_hnsw(
        self, query: str, 
        vector: Optional[List[float]] = None, 
        index: Optional[str] = None, *, k: int = 10, 
        filter_source: Optional[str] = None
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
        rrf_k: int = 60
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
            )
        return [str(h.product_id) for h in hits]