import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

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
            hit: SearchHit = {
                "product_id": src.get("product_id") or h.get("_id"),
                "score": float(h.get("_score", 0.0)),
                "source": src.get("source"),
                "metadata": src.get("metadata") or {},
            }
            if include_full_text:
                hit["full_text"] = src.get("full_text")
            hits.append(hit)
        return hits
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

    def query_hnsw(
        self,
        query: str,
        index: Optional[str] = None,
        *,
        k: int = 10,
        filter_source: Optional[str] = None,
        normalize_embeddings: bool = True,
        include_full_text: bool = False,
    ) -> List[SearchHit]:

        if index is None:
            index = self.hnsw_index

        vec = self.embedding_model.encode(
            query,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        ).tolist()

        body = self.hnsw_query_body(vec, k, filter_source=filter_source)
        res = self.client.search(index=index, body=body)
        hits_raw = res.get("hits", {}).get("hits", [])

        hits: List[SearchHit] = []
        for h in hits_raw:
            src = h.get("_source") or {}
            hit: SearchHit = {
                "product_id": src.get("product_id") or h.get("_id"),
                "score": float(h.get("_score", 0.0)),
                "source": src.get("source"),
                "metadata": src.get("metadata") or {},
            }
            if include_full_text:
                hit["full_text"] = src.get("full_text")
            hits.append(hit)
        return hits

    def query_hnsw_with_vector(
        self,
        vector: List[float],
        index: Optional[str] = None,
        *,
        k: int = 10,
        filter_source: Optional[str] = None,
        normalize_embeddings: bool = True,
        include_full_text: bool = False,
    ) -> List[SearchHit]:

        if index is None:
            index = self.hnsw_index

        body = self.hnsw_query_body(vector, k, filter_source=filter_source)
        res = self.client.search(index=index, body=body)
        hits_raw = res.get("hits", {}).get("hits", [])

        hits: List[SearchHit] = []
        for h in hits_raw:
            src = h.get("_source") or {}
            hit: SearchHit = {
                "product_id": src.get("product_id") or h.get("_id"),
                "score": float(h.get("_score", 0.0)),
                "source": src.get("source"),
                "metadata": src.get("metadata") or {},
            }
            if include_full_text:
                hit["full_text"] = src.get("full_text")
            hits.append(hit)
        return hits

    def ranked_ids_bm25(self, query: str, index: Optional[str] = None, *, k: int = 10, filter_source: Optional[str] = None) -> List[str]:
        return [str(h["product_id"]) for h in self.query_bm25(query, index=index, k=k, filter_source=filter_source)]

    def ranked_ids_hnsw(self, query: str, index: Optional[str] = None, *, k: int = 10, filter_source: Optional[str] = None) -> List[str]:
        return [str(h["product_id"]) for h in self.query_hnsw(query, index=index, k=k, filter_source=filter_source)]
    
    def ranked_ids_hnsw_with_vector(self, vector: List[float], index: Optional[str] = None, *, k: int = 10, filter_source: Optional[str] = None) -> List[str]:
        return [str(h["product_id"]) for h in self.query_hnsw_with_vector(vector, index=index, k=k, filter_source=filter_source)]