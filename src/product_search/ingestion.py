"""
src/product_search/ingestion.py

OpenSearch ingestion layer for the product search project.

SOLID design:
  S — Each class has one job (index lifecycle vs. document building vs. config).
  O — New index types extend BaseIndexer without touching existing classes.
  L — BM25Indexer / HNSWIndexer are interchangeable wherever BaseIndexer is expected.
  I — BaseIndexer exposes only the two methods callers actually need.
  D — HNSWIndexer receives its SentenceTransformer via constructor injection.

Module-level shim functions (create_bm25_index, bulk_ingest_bm25, …) preserve
the API used by database_ingestion.ipynb; new code should use the classes directly.
"""

import argparse
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import torch
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


# ── Utilities ──────────────────────────────────────────────────────────────────

def load_embedder(model_name: str) -> Tuple[SentenceTransformer, str]:
    """
    Load a SentenceTransformer on the best available device.
    Priority: Apple MPS > CUDA > CPU.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return SentenceTransformer(model_name, device=device), device


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ── Document builder ───────────────────────────────────────────────────────────

class DocumentBuilder:
    """
    Single Responsibility: translate a product_store entry into an
    OpenSearch-ready document dict.

    Reads the pre-computed ``full_text`` field written by
    ``ProductDocument.to_dict()`` so text construction is always consistent
    with the data-curation layer — no reimplementation here.
    """

    @staticmethod
    def _base(pid: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        raw = meta.get("metadata", {}) or {}
        # Normalise bullets: collapse pipe/newline delimiters used by ESCI
        bullets_raw = raw.get("product_bullet_point")
        bullets = re.sub(r"[\|\n\r]+", " ", str(bullets_raw)).strip() if bullets_raw else None
        return {
            "product_id":     pid,
            "source":         meta.get("source", ""),
            # Per-field text columns for multi_match boosting
            "title":          raw.get("product_title") or raw.get("product_name"),
            "brand_text":     raw.get("product_brand"),
            "bullets":        bullets,
            "description":    raw.get("product_description"),
            # Full concatenation kept for display; encode_text used for dense embeddings
            "full_text":      meta.get("full_text", ""),
            "encode_text":    meta.get("encode_text", ""),
            # Keyword / numeric filter fields
            "brand":          meta.get("brand"),
            "color":          meta.get("color", []),
            "product_class":  meta.get("product_class"),
            "average_rating": meta.get("average_rating"),
            "review_count":   meta.get("review_count"),
        }

    @staticmethod
    def for_bm25(pid: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        return DocumentBuilder._base(pid, meta)

    @staticmethod
    def for_hnsw(
        pid: str,
        meta: Dict[str, Any],
        vec: List[float],
        vector_field: str,
    ) -> Dict[str, Any]:
        doc = DocumentBuilder._base(pid, meta)
        doc[vector_field] = vec
        return doc


# ── Abstract base ──────────────────────────────────────────────────────────────

class BaseIndexer(ABC):
    """
    Open/Closed + Liskov: common contract for all OpenSearch product indexers.
    New index types (sparse, hybrid, …) extend this without touching existing code.
    """

    def __init__(self, index_name: str) -> None:
        self.index_name = index_name

    @abstractmethod
    def create_index(self, client: OpenSearch) -> None:
        """Create the OpenSearch index if it does not already exist."""

    @abstractmethod
    def bulk_ingest(
        self,
        client: OpenSearch,
        product_store: Dict[str, Dict[str, Any]],
        product_ids: List[str],
    ) -> None:
        """Ingest product_ids from product_store into the index."""


# ── BM25 indexer ──────────────────────────────────────────────────────────────

class BM25Indexer(BaseIndexer):
    """
    Single Responsibility: manages the full lifecycle of a BM25 inverted index.
    Maps all ProductDocument fields so the index is useful for both retrieval
    and filtering (keyword fields for brand, color, product_class, etc.).
    """

    _MAPPINGS = {
        "properties": {
            "product_id":     {"type": "keyword"},
            "source":         {"type": "keyword"},
            "title":          {"type": "text"},
            "brand_text":     {"type": "text"},
            "bullets":        {"type": "text"},
            "description":    {"type": "text"},
            "full_text":      {"type": "text"},
            "brand":          {"type": "keyword"},
            "color":          {"type": "keyword"},
            "product_class":  {"type": "keyword"},
            "average_rating": {"type": "float"},
            "review_count":   {"type": "integer"},
        }
    }

    def __init__(self, index_name: str, batch_size: int = 1000) -> None:
        super().__init__(index_name)
        self.batch_size = batch_size

    def create_index(self, client: OpenSearch) -> None:
        if client.indices.exists(index=self.index_name):
            return
        body = {
            "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": self._MAPPINGS,
        }
        client.indices.create(index=self.index_name, body=body)

    def bulk_ingest(
        self,
        client: OpenSearch,
        product_store: Dict[str, Dict[str, Any]],
        product_ids: List[str],
    ) -> None:
        with tqdm(total=len(product_ids), desc=f"BM25 ingest → {self.index_name}", unit="docs") as pbar:
            for chunk in _batched(product_ids, self.batch_size):
                actions = []
                for pid in chunk:
                    meta = product_store.get(pid)
                    if meta is None:
                        continue
                    actions.append({
                        "_op_type": "index",
                        "_index":   self.index_name,
                        "_id":      pid,
                        "_source":  DocumentBuilder.for_bm25(pid, meta),
                    })
                if actions:
                    helpers.bulk(client, actions)
                pbar.update(len(chunk))
        client.indices.refresh(index=self.index_name)


# ── HNSW configuration ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HNSWConfig:
    """
    Single Responsibility: immutable container for HNSW index parameters.
    Separating config from behaviour keeps HNSWIndexer focused on logic only.
    """
    dim: int
    vector_field: str    = "embedding"
    space_type: str      = "cosinesimil"
    engine: str          = "faiss"
    m: int               = 16
    ef_construction: int = 128
    ef_search: int       = 100


# ── HNSW indexer ──────────────────────────────────────────────────────────────

class HNSWIndexer(BaseIndexer):
    """
    Single Responsibility: manages the full lifecycle of a dense-vector HNSW index.
    Dependency Inversion: SentenceTransformer is injected via constructor so callers
    control model loading (device, fine-tuned checkpoint, etc.).
    """

    def __init__(
        self,
        index_name: str,
        config: HNSWConfig,
        embedder: Optional[SentenceTransformer] = None,
        embedding_service_url: Optional[str] = None,
        batch_size: int = 256,
        encode_batch_size: int = 64,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__(index_name)
        self._config                = config
        self._embedder              = embedder
        self._embedding_service_url = embedding_service_url
        self._batch_size            = batch_size
        self._encode_batch_size     = encode_batch_size
        self._normalize_embeddings  = normalize_embeddings

    def create_index(self, client: OpenSearch) -> None:
        if client.indices.exists(index=self.index_name):
            return
        cfg = self._config
        body = {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "knn.algo_param.ef_search": cfg.ef_search,
                }
            },
            "mappings": {
                "properties": {
                    "product_id":     {"type": "keyword"},
                    "source":         {"type": "keyword"},
                    "full_text":      {"type": "text"},
                    "encode_text":    {"type": "text"},
                    "brand":          {"type": "keyword"},
                    "color":          {"type": "keyword"},
                    "product_class":  {"type": "keyword"},
                    "average_rating": {"type": "float"},
                    "review_count":   {"type": "integer"},
                    cfg.vector_field: {
                        "type":      "knn_vector",
                        "dimension": cfg.dim,
                        "method": {
                            "name":       "hnsw",
                            "space_type": cfg.space_type,
                            "engine":     cfg.engine,
                            "parameters": {
                                "ef_construction": cfg.ef_construction,
                                "m":               cfg.m,
                            },
                        },
                    },
                }
            },
        }
        client.indices.create(index=self.index_name, body=body)

    def bulk_ingest(
        self,
        client: OpenSearch,
        product_store: Dict[str, Dict[str, Any]],
        product_ids: List[str],
    ) -> None:
        if self._embedder is None and not self._embedding_service_url:
            raise RuntimeError(
                "HNSWIndexer.bulk_ingest requires either embedder= or embedding_service_url=. "
                "Pass one to the constructor."
            )
        cfg = self._config
        with tqdm(total=len(product_ids), desc=f"HNSW ingest → {self.index_name}", unit="docs") as pbar:
            for chunk in _batched(product_ids, self._batch_size):
                valid: List[Tuple[str, str, Dict[str, Any]]] = [
                    (pid, meta.get("encode_text") or meta.get("full_text", ""), meta)
                    for pid in chunk
                    if (meta := product_store.get(pid)) is not None
                ]
                if not valid:
                    pbar.update(len(chunk))
                    continue

                texts = [t for _, t, _ in valid]
                if self._embedder is not None:
                    vecs: List[List[float]] = self._embedder.encode(
                        texts,
                        batch_size=min(self._encode_batch_size, len(texts)),
                        show_progress_bar=False,
                        normalize_embeddings=self._normalize_embeddings,
                        convert_to_numpy=True,
                    ).tolist()
                else:
                    resp = requests.post(
                        f"{self._embedding_service_url.rstrip('/')}/encode",
                        json={"texts": texts, "normalize": self._normalize_embeddings},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    vecs = resp.json()["embeddings"]

                actions = []
                for (pid, _, meta), vec in zip(valid, vecs):
                    if len(vec) != cfg.dim:
                        raise ValueError(
                            f"Embedding dimension mismatch for {pid}: "
                            f"got {len(vec)}, expected {cfg.dim}"
                        )
                    actions.append({
                        "_op_type": "index",
                        "_index":   self.index_name,
                        "_id":      pid,
                        "_source":  DocumentBuilder.for_hnsw(pid, meta, vec, cfg.vector_field),
                    })
                if actions:
                    helpers.bulk(client, actions)
                pbar.update(len(chunk))
        client.indices.refresh(index=self.index_name)


# ── Backward-compatible shim functions ────────────────────────────────────────
# Thin wrappers that preserve the function-level API used by database_ingestion.ipynb.
# New code should use BM25Indexer / HNSWIndexer directly.

def create_bm25_index(client: OpenSearch, index_name: str) -> None:
    BM25Indexer(index_name).create_index(client)


def bulk_ingest_bm25(
    client: OpenSearch,
    *,
    index_name: str,
    product_store: Dict[str, Dict[str, Any]],
    product_ids: List[str],
    batch_size: int = 1000,
) -> None:
    BM25Indexer(index_name, batch_size=batch_size).bulk_ingest(client, product_store, product_ids)


def create_hnsw_index(
    client: OpenSearch,
    index_name: str,
    *,
    dim: int,
    vector_field: str    = "embedding",
    space_type: str      = "cosinesimil",
    engine: str          = "faiss",
    m: int               = 16,
    ef_construction: int = 128,
    ef_search: int       = 100,
) -> None:
    config = HNSWConfig(
        dim=dim, vector_field=vector_field, space_type=space_type,
        engine=engine, m=m, ef_construction=ef_construction, ef_search=ef_search,
    )
    HNSWIndexer(index_name, config=config).create_index(client)


def bulk_ingest_hnsw(
    client: OpenSearch,
    *,
    index_name: str,
    product_store: Dict[str, Dict[str, Any]],
    product_ids: List[str],
    embedder: Optional[SentenceTransformer] = None,
    embedding_service_url: Optional[str] = None,
    dim: int,
    vector_field: str       = "embedding",
    space_type: str         = "cosinesimil",
    engine: str             = "faiss",
    batch_size: int         = 256,
    encode_batch_size: int  = 64,
    normalize_embeddings: bool = True,
) -> None:
    config = HNSWConfig(dim=dim, vector_field=vector_field, space_type=space_type, engine=engine)
    HNSWIndexer(
        index_name, config=config, embedder=embedder,
        embedding_service_url=embedding_service_url,
        batch_size=batch_size, encode_batch_size=encode_batch_size,
        normalize_embeddings=normalize_embeddings,
    ).bulk_ingest(client, product_store, product_ids)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest products into OpenSearch.")
    ap.add_argument("--opensearch-host",  default=os.getenv("OPENSEARCH_HOST", "localhost"))
    ap.add_argument("--opensearch-port",  type=int, default=int(os.getenv("OPENSEARCH_PORT", "9200")))
    ap.add_argument("--use-ssl",          action="store_true", default=False)
    ap.add_argument("--verify-certs",     action="store_true", default=False)
    ap.add_argument("--timeout",          type=int, default=int(os.getenv("OPENSEARCH_TIMEOUT", "60")))
    ap.add_argument("--train-qrels",      required=True, help="Path to train_qrels.json")
    ap.add_argument("--test-qrels",       default=None,  help="Path to test_qrels.json (optional — ensures test-relevant products are indexed)")
    ap.add_argument("--product-store",    required=True, help="Path to product_store.json")
    ap.add_argument("--bm25-index",       default=os.getenv("OPENSEARCH_BM25_INDEX", "products_bm25"))
    ap.add_argument("--hnsw-index",       default=os.getenv("OPENSEARCH_HNSW_INDEX", "products_hnsw"))
    ap.add_argument("--vector-field",     default="embedding")
    ap.add_argument("--embedding-service-url", default=os.getenv("CLOUDRUN_URL") or os.getenv("EMBEDDING_SERVICE_URL"), help="Cloud Run embedding service URL (alternative to loading a local model)")
    ap.add_argument("--model",            default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size",        type=int, default=1000)
    ap.add_argument("--hnsw-batch-size",   type=int, default=256)
    ap.add_argument("--encode-batch-size", type=int, default=64)
    ap.add_argument("--no-hnsw",           action="store_true", default=False)
    ap.add_argument("--recreate-indices",  action="store_true", default=False)
    args = ap.parse_args()

    client = OpenSearch(
        hosts=[{"host": args.opensearch_host, "port": args.opensearch_port}],
        use_ssl=args.use_ssl,
        verify_certs=args.verify_certs,
        ssl_show_warn=False,
        timeout=args.timeout,
    )

    with open(args.product_store, encoding="utf-8") as f:
        product_store: Dict[str, Dict[str, Any]] = json.load(f)

    with open(args.train_qrels, encoding="utf-8") as f:
        train_qrels: Dict[str, Dict[str, float]] = json.load(f)

    all_qrels = dict(train_qrels)
    if args.test_qrels:
        with open(args.test_qrels, encoding="utf-8") as f:
            test_qrels: Dict[str, Dict[str, float]] = json.load(f)
        all_qrels.update(test_qrels)
        print(f"[INFO] Merging train + test qrels: {len(train_qrels)} + {len(test_qrels)} queries")

    product_ids = sorted({pid for gains in all_qrels.values() for pid in gains})
    missing = [pid for pid in product_ids if pid not in product_store]
    if missing:
        print(f"[WARN] {len(missing)} product_ids from qrels missing in product_store. Examples: {missing[:5]}")

    if args.recreate_indices:
        targets = [args.bm25_index] + ([] if args.no_hnsw else [args.hnsw_index])
        for idx in targets:
            if client.indices.exists(index=idx):
                client.indices.delete(index=idx)

    t0 = time.time()
    bm25 = BM25Indexer(args.bm25_index, batch_size=args.batch_size)
    bm25.create_index(client)
    print(f"[OK] BM25 index ready: {args.bm25_index}")
    bm25.bulk_ingest(client, product_store, product_ids)
    print(f"[OK] BM25 ingested: {args.bm25_index} "
          f"(docs={client.count(index=args.bm25_index)['count']}, {time.time()-t0:.1f}s)")

    if not args.no_hnsw:
        emb_url = args.embedding_service_url
        if emb_url:
            print(f"[INFO] Using Cloud Run embedding service: {emb_url}")
            print(f"[INFO] Probing embedding service (may take up to 120s on cold start)...")
            probe = requests.post(
                f"{emb_url.rstrip('/')}/encode",
                json={"texts": ["probe"], "normalize": True},
                timeout=120,
            )
            probe.raise_for_status()
            dim = len(probe.json()["embeddings"][0])
            embedder = None
            print(f"[INFO] Embedding dim from service: {dim}")
        else:
            embedder, device = load_embedder(args.model)
            print(f"[INFO] SentenceTransformer device = {device}")
            dim = int(embedder.get_sentence_embedding_dimension())
        config = HNSWConfig(dim=dim, vector_field=args.vector_field, engine="faiss")
        hnsw = HNSWIndexer(
            args.hnsw_index, config=config, embedder=embedder,
            embedding_service_url=emb_url,
            batch_size=args.hnsw_batch_size, encode_batch_size=args.encode_batch_size,
        )
        hnsw.create_index(client)
        print(f"[OK] HNSW index ready: {args.hnsw_index} (dim={dim})")
        hnsw.bulk_ingest(client, product_store, product_ids)
        print(f"[OK] HNSW ingested: {args.hnsw_index} "
              f"(docs={client.count(index=args.hnsw_index)['count']})")

    print("[DONE] Ingestion complete.")


if __name__ == "__main__":
    main()
