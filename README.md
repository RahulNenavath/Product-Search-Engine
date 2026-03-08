# Product Search Engine

A modular **product search engine** built to experiment with and benchmark **retrieval + ranking** strategies for e-commerce search. The goal is to identify configurations that return **highly relevant products** for natural-language queries, balancing **precision, recall, and latency**.

The project benchmarks methods using two publicly available product-search datasets:
- **Amazon ESCI** — large-scale shopping queries with graded relevance labels
- **Wayfair WANDS** — furniture/home-goods queries with human annotations

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Datasets](#datasets)
3. [Unified Product Schema](#unified-product-schema)
4. [Data Curation](#data-curation)
5. [Search Engine Architecture](#search-engine-architecture)
6. [Encoder Fine-tuning](#encoder-fine-tuning)
7. [Evaluation and Metrics](#evaluation-and-metrics)
8. [Results](#results)
9. [System Components and Docker](#system-components-and-docker)
10. [Quickstart](#quickstart)
11. [Repository Structure](#repository-structure)

---

## Problem Statement

In e-commerce, search quality directly impacts discovery and conversion. Given a user query (often short, ambiguous, and noisy), the system must retrieve a small set of candidate products and rank them so that the most relevant items appear at the top.

This repository implements and compares multiple retrieval approaches:

1. **BM25** (lexical retrieval) — strong baseline for exact term matches, brand names, and structured tokens.
2. **Dense retrieval (HNSW k-NN)** over learned embeddings — improves recall for paraphrases and intent-level matches beyond keyword overlap.
3. **Hybrid retrieval** (BM25 + HNSW) with rank aggregation — **Reciprocal Rank Fusion (RRF)** to combine lexical + semantic candidates.
4. **Hybrid + Cross-encoder reranking** — refines the final ordering over a candidate pool using a stronger interaction model.

---

## Datasets

### Wayfair WANDS

**WANDS** is a curated product search relevance dataset for e-commerce benchmarking. It contains **42,994 candidate products**, **480 queries**, and **233,448 (query, product) relevance judgements**. [Source](https://github.com/wayfair/WANDS)

**Files and schema (CSV):**
- `product.csv` — `product_id`, `product_name`, `product_class`, `category_hierarchy`, `product_description`, `product_features` (pipe-delimited attribute:value pairs), `rating_count`, `average_rating`, `review_count`
- `query.csv` — `query_id`, `query`, `query_class`
- `label.csv` — `id`, `query_id`, `product_id`, `label` ∈ {`Exact`, `Partial`, `Irrelevant`}

### Amazon Shopping Queries (ESCI)

Amazon's **ESCI** dataset is a large-scale benchmark for semantic matching between shopping queries and products. Each query has up to 40 candidate results with a graded 4-label relevance scheme. [Source](https://github.com/amazon-science/esci-data)

**Relevance labels:**
- **Exact (E):** satisfies all query specifications → gain `3.0`
- **Substitute (S):** partially satisfies the query → gain `2.0`
- **Complement (C):** used alongside an exact match → gain `1.0`
- **Irrelevant (I):** unrelated → gain `0.0` (excluded from qrels by default)

**Dataset size:** ~48,300 unique queries, ~1.1M judgements (small version, Task 1).

---

## Unified Product Schema

To support both BM25 and HNSW retrieval consistently, all products from both datasets are normalized into a single OpenSearch document schema defined in [`src/product_search/data_curation.py`](src/product_search/data_curation.py):

| Field | OpenSearch Type | Notes |
|---|---|---|
| `product_id` | keyword | Prefixed: `amz_<id>` / `wands_<id>` |
| `source` | keyword | `"ESCI"` or `"WANDS"` |
| `full_text` | text | Enriched text: title → brand → bullets → description |
| `brand` | keyword | Normalised, lowercased |
| `color` | keyword[] | Normalised list (splits compound colour strings) |
| `product_class` | keyword | High-level category |
| `category_path` | keyword | WANDS hierarchy; null for ESCI |
| `average_rating` | float | WANDS only |
| `review_count` | integer | WANDS only |
| `metadata` | object | Full raw product fields, preserved for debugging |
| `embedding` | knn_vector | Dense vector (HNSW index only) |

**`full_text` construction** concatenates the highest-signal fields in priority order:
- ESCI: `product_title` → `product_brand` → `product_bullet_point` → `product_description`
- WANDS: `product_name` → `product_class` → `category_hierarchy` → feature values → `product_description`

---

## Data Curation

Data curation is handled by [`src/product_search/data_curation.py`](src/product_search/data_curation.py), which follows a SOLID design:

- **`ESCIProcessor`** — normalises raw Amazon ESCI DataFrames into `DatasetArtifacts`
- **`WANDSProcessor`** — normalises raw Wayfair WANDS DataFrames into `DatasetArtifacts`
- **`DatasetMerger`** — merges any number of processors into a single `MergedArtifacts` object

```python
from product_search.data_curation import ESCIProcessor, WANDSProcessor, DatasetMerger

esci  = ESCIProcessor(amz_train_df, amz_test_df)
wands = WANDSProcessor(wands_train_df, wands_test_df)

merger    = DatasetMerger([esci, wands])
artifacts = merger.build()
# artifacts.product_store       → {product_id: dict}
# artifacts.train_qrels_dict    → {query_id: {product_id: gain}}
# artifacts.test_qrels_dict     → same
# artifacts.train_query_table   → pd.DataFrame [query_id, query]
# artifacts.test_query_table    → pd.DataFrame [query_id, query]
```

### Development Sample

The full dataset (~525k documents) is large for local development. `DatasetMerger.build_dev_sample()` builds a reproducible small sample:

- **2,000 train queries** (1,000 ESCI + 1,000 WANDS)
- **500 test queries** (250 ESCI + 250 WANDS)
- **Product store** limited to queried products + 500 distractor products (250 per source)

```python
artifacts = merger.build_dev_sample(
    n_train_per_source=1000,
    n_test_per_source=250,
    n_distractor_per_source=250,
    random_state=42,
)
```

The full pipeline is documented in [`notebooks/database_ingestion.ipynb`](notebooks/database_ingestion.ipynb).

---

## Search Engine Architecture

All retrieval strategies share a common request/response contract via `OpenSearchInference` in [`src/product_search/search_pipeline.py`](src/product_search/search_pipeline.py):

- **Input:** `query` (str), `k` (int), optional `filter_source` (`"ESCI"` | `"WANDS"`)
- **Output:** ranked `SearchHit` list — `product_id`, `score`, `source`, `full_text`, `metadata`

Two OpenSearch indices are maintained independently:

| Index | Type | Purpose |
|---|---|---|
| `products_bm25` | Inverted text index | Lexical keyword search |
| `products_hnsw` | k-NN vector index | Dense semantic search |

### Strategy A — BM25 (Lexical)

Runs a `match` query over the `full_text` field with an optional `source` filter.

```
query → match on full_text → top-k hits (sorted by BM25 score)
```

### Strategy B — HNSW k-NN (Dense / Semantic)

1. Encodes the user query with the fine-tuned `SentenceTransformer`.
2. Issues a `knn` query against the `embedding` field.
3. Optionally applies a `source` filter.

```
query → encode → knn on embedding → top-k hits
```

### Strategy C — Hybrid via Reciprocal Rank Fusion (RRF)

Combines lexical and semantic signals without learning-to-rank:

1. Retrieve a candidate pool of size `C` from BM25.
2. Retrieve a candidate pool of size `C` from HNSW.
3. Fuse ranked lists with **RRF**: `score(d) = Σ 1 / (rrf_k + rank(d))`
4. Return top-k fused documents.

Default hyperparameters: `C = 20`, `rrf_k = 60`.

### Strategy D — Hybrid + Cross-Encoder Reranking

Maximises ranking quality using a stronger model over a small candidate set:

1. Retrieve top `C` from BM25 and top `C` from HNSW.
2. Union + deduplicate candidates by `product_id`.
3. Build `(query, full_text)` pairs and score with a `CrossEncoder`.
4. Return top-k reranked documents.

---

## Encoder Fine-tuning

Dense retrieval quality depends on how well the encoder maps **(query, product)** text into a shared embedding space. Off-the-shelf encoders are not optimised for product-search relevance, so this project fine-tunes a `SentenceTransformer` on dataset-specific training pairs.

### Training Objective

A bi-encoder is fine-tuned so that:
- the **query embedding** is close to **relevant product** embeddings (positives),
- and far from **non-relevant product** embeddings (negatives).

### Data Construction

For each query, the **highest-gain judged product** is selected as the positive. The product text is built using `FullTextBuilder`, consistent with indexing.

### Loss Function: CachedGISTEmbedLoss

Uses a guide/teacher model (`BAAI/bge-m3`) to identify likely false negatives during in-batch contrastive training, reducing noise without explicit negative mining.

### Configuration

- **Base model:** `all-MiniLM-L6-v2`
- **Guide model:** `BAAI/bge-m3`
- Training via `SentenceTransformerTrainer` with mixed precision (`fp16=True`)
- Requires an NVIDIA GPU (CUDA)

The fine-tuned model is saved to `src/product_search/finetuned_encoder/` and downloaded from S3 using [`src/product_search/download_model_from_s3.py`](src/product_search/download_model_from_s3.py):

```bash
python src/product_search/download_model_from_s3.py --profile <aws-profile>
```

---

## Evaluation and Metrics

Evaluation uses held-out **test sets** with metrics computed at cutoffs **K ∈ {5, 10, 20}**.

### Recall@K
Fraction of all relevant items for a query that appear within the top-K results. Higher Recall@K = better candidate coverage — critical for two-stage systems where rerankers need good candidates.

### NDCG@K (Normalized Discounted Cumulative Gain)
Measures ranking quality with **position discounting** and **graded relevance**. Items at rank 1 contribute more than items at rank 10. Normalised against the ideal ranking (IDCG). **Most informative metric** for user-facing search.

### MRR@K (Mean Reciprocal Rank)
Average of `1 / rank` of the first relevant result across all queries. Strong proxy for "time-to-success" on navigational or single-intent queries.

---

## Results

### NDCG@K on Test Set

| Configuration | NDCG@5 | NDCG@10 | NDCG@20 |
|---|:---:|:---:|:---:|
| BM25 | 0.37 | 0.33 | 0.32 |
| Base Encoder (HNSW) | 0.32 | 0.29 | 0.29 |
| Fine-tuned Encoder (HNSW) | 0.37 | 0.34 | 0.33 |
| Hybrid Fine-tuned + RRF | 0.40 | 0.36 | 0.36 |
| **Hybrid Fine-tuned + Cross-Encoder** | **0.47** | **0.42** | **0.41** |

### Key Findings

1. **Fine-tuning improves dense retrieval materially.** The fine-tuned encoder matches BM25 at NDCG@5 (0.37 vs 0.37), confirming the encoder better captures domain relevance.
2. **Hybrid + RRF is a strong, simple improvement.** Reaching 0.40 NDCG@5 without any additional learned components — BM25 and HNSW retrieve complementary relevant items.
3. **Hybrid + Cross-Encoder is the best configuration** at every cutoff (0.47 / 0.42 / 0.41), confirming the value of a two-stage architecture: maximize recall in stage 1, maximize precision in stage 2.

---

## System Components and Docker

The system runs as three Docker services orchestrated by `docker-compose.yml`:

```
┌────────────────────────────────────────────────────────────────┐
│                      docker-compose                            │
│                                                                │
│  ┌──────────────┐   ┌─────────────────┐   ┌────────────────┐  │
│  │  opensearch  │   │      api        │   │      ui        │  │
│  │  :9200       │◄──│  FastAPI :8000  │◄──│  Streamlit     │  │
│  │  BM25 index  │   │  /search/bm25   │   │  :8501         │  │
│  │  HNSW index  │   │  /search/hnsw   │   │  Search UI     │  │
│  └──────────────┘   │  /search/hybrid │   └────────────────┘  │
│                     └─────────────────┘                        │
└────────────────────────────────────────────────────────────────┘
```

### Service: `opensearch`
- Image: `opensearchproject/opensearch:3.4.0`
- Hosts both the BM25 inverted index and the HNSW k-NN vector index
- Data persisted in a named Docker volume (`opensearch-data`)
- Exposed at `localhost:9200`

### Service: `api`
- Built from [`api/Dockerfile`](api/Dockerfile) using the project root as context
- FastAPI application ([`api/main.py`](api/main.py)) backed by [`src/product_search/search_pipeline.py`](src/product_search/search_pipeline.py)
- Loads the fine-tuned encoder from `src/product_search/finetuned_encoder/` (volume-mounted read-only into the container at `/models/finetuned_encoder`)
- Exposed at `localhost:8000`

**API endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check + OpenSearch connectivity |
| `POST` | `/search/bm25` | BM25 keyword search |
| `POST` | `/search/hnsw` | Dense semantic search (fine-tuned encoder) |
| `POST` | `/search/hybrid` | Hybrid BM25 + HNSW via RRF |

**Request body** (all search endpoints):
```json
{
  "query": "red running shoes for women",
  "k": 10,
  "filter_source": "ESCI"
}
```
`filter_source` is optional — omit to search across both ESCI and WANDS.

### Service: `ui`
- Built from [`ui/Dockerfile`](ui/Dockerfile)
- Streamlit application ([`ui/app.py`](ui/app.py)) that calls the `api` service internally via `http://api:8000`
- Exposes a visual search interface at `localhost:8501`
- Features: search mode toggle (BM25 / HNSW / Hybrid), `k` slider, source filter, live API health check, results as expandable cards

---

## Quickstart

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- The fine-tuned encoder model in `src/product_search/finetuned_encoder/`
  - Download it from S3: `python src/product_search/download_model_from_s3.py --profile <aws-profile>`
  - Or skip this if you want to use a HuggingFace model instead (set `EMBEDDING_MODEL_PATH` in `dev.env`)

### Step 1 — Environment variables

Copy the example env file and fill in values:

```bash
cp .env.example dev.env
```

Minimum required values in `dev.env`:

```env
OPENSEARCH_BM25_INDEX=products_bm25
OPENSEARCH_HNSW_INDEX=products_hnsw
```

All other variables have sensible defaults (see [`api/main.py`](api/main.py) for the full list).

### Step 2 — Start all services

```bash
docker compose up --build
```

This starts OpenSearch, waits for it to be healthy, then starts the API and UI.

| Service | URL |
|---|---|
| Search UI (Streamlit) | http://localhost:8501 |
| API (FastAPI) | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| OpenSearch | http://localhost:9200 |

### Step 3 — Ingest data

Before searching, you need to index products into OpenSearch. Run the ingestion notebook:

```
notebooks/database_ingestion.ipynb
```

This notebook:
1. Loads and curates the ESCI and WANDS datasets using `DatasetMerger`
2. Creates the BM25 index and ingests all products
3. Creates the HNSW index and ingests all product embeddings

> For local development on a low resource machine, use `merger.build_dev_sample()` instead of `merger.build()` to index a smaller ~5k product subset (or even lower based on resource availability).

### Step 4 — Search

Open `http://localhost:8501` to use the Streamlit search UI, or call the API directly:

```bash
# BM25 search
curl -X POST http://localhost:8000/search/bm25 \
  -H "Content-Type: application/json" \
  -d '{"query": "red running shoes", "k": 5}'

# Hybrid search
curl -X POST http://localhost:8000/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "red running shoes", "k": 5}'
```

### (Optional) Local Python environment

```bash
bash setup.sh
conda activate product_search_env
```

---

> **Summary:** Hybrid retrieval with a fine-tuned encoder + cross-encoder reranking is the top-performing configuration (NDCG@5 = 0.47). The system runs end-to-end in Docker — OpenSearch stores both the BM25 and HNSW indices, the FastAPI service exposes three search endpoints, and the Streamlit UI provides a visual interface for exploration.
