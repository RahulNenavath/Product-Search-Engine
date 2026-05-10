# Product Search Engine

A modular **product search engine** built to experiment with and benchmark **retrieval + ranking** strategies for e-commerce search. The goal is to identify configurations that return **highly relevant products** for natural-language queries, balancing **precision, recall, and latency**.

The project benchmarks methods using two publicly available product-search datasets:

- **Amazon ESCI** — large-scale shopping queries with graded relevance labels
- **Wayfair WANDS** — furniture/home-goods queries with human annotations

ML models (encoder + reranker) are served from a **GPU-backed Cloud Run service on GCP**, loaded at runtime from a **GCS bucket** — no models are baked into Docker images.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Datasets](#datasets)
3. [Unified Product Schema](#unified-product-schema)
4. [Data Curation](#data-curation)
5. [Search Engine Architecture](#search-engine-architecture)
6. [Encoder Fine-tuning](#encoder-fine-tuning)
7. [Agentic Search Architecture](#agentic-search-architecture)
8. [Evaluation and Metrics](#evaluation-and-metrics)
9. [Results](#results)
10. [System Components and Docker](#system-components-and-docker)
11. [GCP Setup — Embedding Service](#gcp-setup--embedding-service)
12. [CI/CD — GitHub Actions](#cicd--github-actions)
13. [Quickstart](#quickstart)
14. [Repository Structure](#repository-structure)

---

## Problem Statement

In e-commerce, search quality directly impacts discovery and conversion. Given a user query (often short, ambiguous, and noisy), the system must retrieve a small set of candidate products and rank them so that the most relevant items appear at the top.

This repository implements and compares multiple retrieval approaches:

1. **BM25** (lexical retrieval) — strong baseline for exact term matches, brand names, and structured tokens.
2. **Dense retrieval (HNSW k-NN)** over learned embeddings — improves recall for paraphrases and intent-level matches beyond keyword overlap.
3. **Hybrid retrieval** (BM25 + HNSW) with rank aggregation — **Reciprocal Rank Fusion (RRF)** to combine lexical + semantic candidates.
4. **Hybrid + Cross-encoder reranking** — refines the final ordering over a candidate pool using a stronger interaction model.
5. **Agentic search** (LangGraph + Gemini Flash) — a LangGraph pipeline that rewrites and normalises the query before retrieval, then assesses result quality and optionally retries with a wider candidate pool or decomposes multi-constraint queries.

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

| Label | Meaning | Gain |
|---|---|---|
| **Exact (E)** | Satisfies all query specifications | `3.0` |
| **Substitute (S)** | Partially satisfies the query | `2.0` |
| **Complement (C)** | Used alongside an exact match | `1.0` |
| **Irrelevant (I)** | Unrelated | `0.0` (excluded from qrels by default) |

**Dataset size:** ~48,300 unique queries, ~1.1M judgements (small version, Task 1).

---

## Unified Product Schema

To support both BM25 and HNSW retrieval consistently, all products from both datasets are normalized into a single OpenSearch document schema defined in [`src/product_search/data_curation.py`](src/product_search/data_curation.py):

| Field | OpenSearch Type | Notes |
|---|---|---|
| `product_id` | keyword | Prefixed: `amz_<id>` / `wands_<id>` |
| `source` | keyword | `"ESCI"` or `"WANDS"` |
| `title` | text | Product title — highest BM25 boost (×5) |
| `brand_text` | text | Brand name — second BM25 boost (×4) |
| `bullets` | text | Bullet-point attributes — moderate boost (×2) |
| `description` | text | Long-form description — down-weighted (×0.3) |
| `full_text` | text | All fields concatenated; used for display |
| `encode_text` | text | Short form for embeddings: title → brand → color |
| `brand` | keyword | Normalised, lowercased |
| `color` | keyword[] | Normalised list (splits compound colour strings) |
| `product_class` | keyword | High-level category |
| `category_path` | keyword | WANDS hierarchy; null for ESCI |
| `average_rating` | float | WANDS only |
| `review_count` | integer | WANDS only |
| `metadata` | object | Full raw product fields, preserved for debugging |
| `embedding` | knn_vector | 384-dim dense vector (HNSW index only) |

**Field construction:**

- `full_text` — ESCI: `product_title` → `product_brand` → `product_bullet_point` → `product_description`. WANDS: `product_name` → `product_class` → `category_hierarchy` → `feature values` → `product_description`
- `encode_text` — short-form representation used for embedding: title → brand → color list. Mirrors the text used during encoder fine-tuning to eliminate train/serve mismatch.

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

Encoding and reranking are **not performed locally** — all calls are proxied to the remote `embedding_service` running on Cloud Run (see [GCP Setup](#gcp-setup--embedding-service)).

### Strategy A — BM25 (Lexical)

```
query → _clean_query() → multi_match (title^5, brand_text^4, bullets^2, description^0.3) → top-k
```

Multi-field BM25 with `best_fields` scoring. Queries are pre-cleaned to strip conversational filler ("I'm looking for…", "show me some great…") before scoring.

### Strategy B — HNSW k-NN (Dense / Semantic)

```
query → POST /encode (embedding_service) → knn on embedding field → top-k hits
```

The encoder is called with `encode_text`-style input (title + brand + color) to match the representation used during fine-tuning.

### Strategy C — Hybrid via Reciprocal Rank Fusion (RRF)

```
query → BM25 top-C  ─┐
                      ├→ RRF fusion → top-k
query → HNSW top-C  ─┘
```

Default hyperparameters: `C = 50` (candidate pool per method), `rrf_k = 60`.

### Strategy D — Hybrid + Cross-Encoder Reranking

```
query → BM25 top-C  ─┐
                      ├→ union/dedup → POST /rerank (embedding_service) → top-k
query → HNSW top-C  ─┘
```

Default hyperparameters: `C = 25` per method (smaller pool → reranker sees higher-precision candidates).

---

## Agentic Search Architecture

The agentic search endpoint (`POST /search/agentic`) wraps `hybrid_rerank` in a **LangGraph** orchestration loop powered by two Gemini Flash models on Vertex AI. The agent's job is purely query understanding and result quality assessment — retrieval always uses `hybrid_rerank`.

### Pipeline Flow

```
raw_query
    │
    ▼
[understand_query]  gemini-3.1-flash-lite-preview
  Rewrites query: strips filler, bridges vocabulary gaps, expands underspecified terms
  Extracts: filter_source (ESCI | WANDS | none)
  Classifies: complexity (simple | complex)
    │
    ▼
[retrieve]          hybrid_rerank (BM25 + HNSW + cross-encoder, pool=25)
  → hits: List[SearchHit]
  → iteration += 1
    │
    ├── complexity == "simple"  ──────────────────────────────────► END
    │
    ▼
[assess_results]    gemini-3-flash-preview
  Judges top-5 results (with reranker scores) against original query
  quality: good | low_coverage | wrong_category | semantic_drift | multi_constraint_miss
  action:  return | widen | decompose
    │
    ├── "return" or iteration ≥ 2  ──────────────────────────────► END
    │
    ├── "widen"  →  candidate_pool_size × 2  ──────────────────► [retrieve]
    │
    └── "decompose"  →  split into 2 sub-queries, retrieve both, merge  ──► END
```

### Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Retrieval strategy | Always `hybrid_rerank` | Removes routing complexity; reranker quality is consistent |
| `understand_query` model | `gemini-3.1-flash-lite-preview` | Fast and cheap; structured output via Pydantic |
| `assess_results` model | `gemini-3-flash-preview` | Stronger reasoning for quality judgement |
| Simple query shortcut | Skip `assess_results` | Saves one LLM call for the majority of queries |
| Max iterations | 2 | Prevents runaway loops; `assess_results` always returns on iteration ≥ 2 |
| Vertex AI region | `global` | Gemini 3 preview models are only available on the global endpoint |
| Structured output | Pydantic with `str = ""` defaults | Vertex AI protobuf rejects `Optional[str]` / `null` types |

### Query Rewriting Rules (IR-grounded)

The `understand_query` prompt applies evidence-based rewriting designed for hybrid BM25 + HNSW retrieval:

- **Pass-through (no rewrite):** brand + model queries (`"braun thermoscan 5"`), navigational queries (person names, product names), garbled text, and service queries (`"amazon rewards points"`)
- **Vocabulary gap bridging:** append the catalog synonym rather than replace (`"couch"` → `"couch sofa"`) so BM25 retains both tokens
- **Expansion:** only for 1–2 word underspecified category queries (`"rug"` → `"area rug carpet"`)
- **Attribute preservation:** all stated constraints (color, material, price signals, negations) must survive verbatim

### Result Assessment (Score-Aware)

`assess_results` receives the top-5 hits with their **cross-encoder reranker scores**, enabling score-threshold reasoning alongside snippet analysis:

- Score ≥ 0.5 → strong match; default to `return`
- Score < 0.1 → retrieval failure; consider `widen` for `low_coverage`
- `widen` is only triggered for `low_coverage` — it does not help `wrong_category` or `semantic_drift`

### Current Performance and Limitations

Benchmarked on the full 200-query test set (100 ESCI + 100 WANDS):

| Metric | hybrid_rerank | agentic (v1) | Δ |
|---|---|---|---|
| NDCG@5 | **0.682** | 0.657 | −3.7% |
| NDCG@10 | **0.661** | 0.635 | −3.9% |
| NDCG@20 | **0.651** | 0.628 | −3.5% |
| MRR@5 | **0.826** | 0.804 | −2.7% |
| Recall@20 | **0.476** | 0.459 | −3.6% |
| Latency | ~0.5 s | ~3.7 s | 7× slower |

The v1 agent slightly underperforms the baseline because the hybrid_rerank pipeline (fine-tuned encoder + cross-encoder reranker) already handles vocabulary gaps, making query rewriting redundant for most queries. The LLM rewrites introduce marginal noise on well-specified queries that outweighs the gains on ambiguous ones.

**Planned v2 architecture** addresses this by removing query rewriting and instead using the LLM for:

1. **Structured filter extraction** — colour and brand constraints become OpenSearch hard filters (not soft query tokens), directly improving precision
2. **Query-type routing** — navigational/brand-model queries take a BM25-only fast path; multi-constraint queries are decomposed before retrieval rather than after
3. **Score-threshold gating** — `assess_results` is replaced by a heuristic: if the top-1 reranker score is below a threshold, widen and retry without any LLM call

---

## Encoder Fine-tuning

Dense retrieval quality depends on how well the encoder maps **(query, product)** text into a shared embedding space. Off-the-shelf encoders are not optimised for product-search relevance, so this project fine-tunes a `SentenceTransformer` on dataset-specific training pairs.

### Training Objective

A bi-encoder is fine-tuned so that the query embedding is close to relevant product embeddings (positives) and far from non-relevant ones (negatives).

### Data Construction

For each query, the **highest-gain judged product** is selected as the positive. Product text is built using `FullTextBuilder`, consistent with indexing.

### Loss Function: CachedGISTEmbedLoss

Uses a guide/teacher model (`BAAI/bge-m3`) to identify likely false negatives during in-batch contrastive training, reducing noise without explicit negative mining.

### Configuration

| Setting | Value |
|---|---|
| Base model | `sentence-transformers/all-MiniLM-L6-v2` |
| Guide model | `BAAI/bge-m3` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Training precision | `fp16` (requires CUDA) |

The fine-tuned model is stored in GCS and served via the Cloud Run embedding service — see [GCP Setup](#gcp-setup--embedding-service).

> **Don't have a fine-tuned model?** The embedding service defaults gracefully to `sentence-transformers/all-MiniLM-L6-v2` from HuggingFace if no fine-tuned checkpoint is present in GCS. See the model upload section below.

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

All metrics are computed on held-out test sets (100 ESCI + 100 WANDS queries), `top-100` retrieval pool, cutoffs K ∈ {5, 10, 20}. Hybrid Rerank results are from a 9,100-query ablation run; all other results from the 200-query test set.

### NDCG@K

| Configuration | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|
| BM25 | 0.6030 | 0.5767 | 0.5530 |
| HNSW (Fine-tuned Encoder) | 0.6335 | 0.6149 | 0.6031 |
| Hybrid RRF (BM25 + HNSW, C=50) | 0.6576 | 0.6373 | 0.6176 |
| **Hybrid + Cross-Encoder Reranker (C=25)** | **0.6824** | **0.6608** | **0.6510** |
| Agentic (Gemini Flash + hybrid_rerank) | 0.6574 | 0.6347 | 0.6282 |

### MRR@K

| Configuration | MRR@5 | MRR@10 | MRR@20 |
|---|---|---|---|
| BM25 | 0.7598 | 0.7671 | 0.7689 |
| HNSW (Fine-tuned Encoder) | 0.8141 | 0.8176 | 0.8186 |
| Hybrid RRF | 0.8418 | 0.8466 | 0.8470 |
| **Hybrid + Cross-Encoder Reranker** | **0.8262** | **0.8305** | **0.8316** |
| Agentic (Gemini Flash + hybrid_rerank) | 0.8043 | 0.8093 | 0.8106 |

### Recall@K

| Configuration | Recall@5 | Recall@10 | Recall@20 |
|---|---|---|---|
| BM25 | 0.1544 | 0.2627 | 0.3896 |
| HNSW (Fine-tuned Encoder) | 0.1662 | 0.2951 | 0.4440 |
| Hybrid RRF | 0.1705 | 0.2976 | 0.4471 |
| **Hybrid + Cross-Encoder Reranker** | **0.1839** | **0.3151** | **0.4758** |
| Agentic (Gemini Flash + hybrid_rerank) | 0.1786 | 0.3060 | 0.4593 |

### Key Findings

1. **Multi-field BM25 with per-field boosting** (`title^5, brand_text^4, bullets^2, description^0.3`) lifts the BM25 baseline from ~0.586 to 0.603 NDCG@5 compared to a single `full_text` field — making it a stronger stage-1 retriever.
2. **Fine-tuned encoder outperforms BM25 at every cutoff**, confirming the encoder better captures domain-level relevance beyond keyword overlap (+5 pp NDCG@5).
3. **Hybrid RRF is a strong, calibration-free improvement** — it reaches NDCG@5 = 0.658 without any additional learned components, a +9 pp gain over BM25.
4. **Hybrid + Cross-Encoder Reranker is the best configuration at all cutoffs** (NDCG@5 = 0.682, +13 pp vs BM25). Candidate pool C=25 was found to be optimal — larger pools feed the reranker more noise than signal.
5. **MRR is highest for Hybrid RRF** (0.847), slightly above the reranker (0.832), suggesting RRF is better at surfacing *a* relevant result at rank 1, while the reranker better handles the full top-K ordering.
6. **Agentic search (v1) scores slightly below the hybrid_rerank baseline** (NDCG@20: 0.628 vs 0.651, −3.5%) despite using the same retrieval backend. LLM query rewriting introduces marginal noise on the ~80% of well-specified queries for which the fine-tuned encoder already handles vocabulary gaps. Latency is 7× higher (~3.7 s vs ~0.5 s). A v2 design replacing rewriting with structured filter extraction and query-type routing is in progress.

> **For production ranking** (k ≤ 20): use **Hybrid + Cross-Encoder Reranker**.
> **For recall-oriented retrieval** (broad candidates, fast latency): use **Hybrid RRF**.
> **For agentic / conversational search**: use the `/search/agentic` endpoint (LangGraph + Gemini Flash); note higher latency and ongoing development.

---

## System Components and Docker

The system runs as three local Docker services orchestrated by `docker-compose.yml`, with a fourth service (embedding) deployed remotely on GCP Cloud Run:

### Service: `opensearch`

- Image: `opensearchproject/opensearch:3.4.0`
- Hosts both the BM25 inverted index and the HNSW k-NN vector index
- Data persisted in a named Docker volume (`opensearch-data`)
- Exposed at `localhost:9200`

### Service: `api`

- Built from [`api/Dockerfile`](api/Dockerfile)
- FastAPI application ([`api/main.py`](api/main.py)) backed by [`src/product_search/search_pipeline.py`](src/product_search/search_pipeline.py)
- Calls the remote embedding service via `EMBEDDING_SERVICE_URL` env var
- Exposed at `localhost:8000`

**API endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check + OpenSearch + embedding service connectivity |
| `POST` | `/search/bm25` | BM25 keyword search |
| `POST` | `/search/hnsw` | Dense semantic search (fine-tuned encoder) |
| `POST` | `/search/hybrid` | Hybrid BM25 + HNSW via RRF |
| `POST` | `/search/hybrid_rerank` | Hybrid RRF + cross-encoder reranking |
| `POST` | `/search/agentic` | LangGraph agent (Gemini Flash) + hybrid_rerank; returns `rewritten_query` field |

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
- Features: search mode toggle (BM25 / HNSW / Hybrid / Hybrid+Rerank / Agentic), `k` slider, source filter, live API health check, results as expandable cards
- Agentic mode shows the rewritten query in a highlighted info box for transparency

### Service: `embedding_service` (GCP Cloud Run)

- GPU-backed FastAPI service that handles all encoding and reranking
- Models are **not baked into the image** — they are mounted at runtime from a GCS bucket
- Auto-deploys on every push to `embedding_service/**` via GitHub Actions
- See [GCP Setup](#gcp-setup--embedding-service) for full provisioning instructions

---

## GCP Setup — Embedding Service

This section walks through setting up the GCP infrastructure from scratch to host the embedding service on Cloud Run with a GPU.

### Prerequisites

- [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) installed and authenticated
- A GCP project with billing enabled
- Docker installed locally (for the one-time model upload)

### Step 1 — Create GCP Project and Enable APIs

```bash
gcloud projects create <YOUR_PROJECT_ID> --name="Product Search"
gcloud config set project <YOUR_PROJECT_ID>

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com
```

### Step 2 — Create Artifact Registry Repository

```bash
gcloud artifacts repositories create <YOUR_REPO_NAME> \
  --repository-format=docker \
  --location=<YOUR_REGION> \
  --description="Product search Docker images"
```

### Step 3 — Create GCS Bucket and Upload Models

Create the bucket (use a region with Cloud Run GPU support, e.g. `<YOUR_REGION>`):

```bash
gcloud storage buckets create gs://<YOUR_BUCKET_NAME> \
  --location=<YOUR_REGION> \
  --uniform-bucket-level-access
```

Upload models to GCS. The script handles both a local fine-tuned model and HuggingFace downloads:

```bash
pip install google-cloud-storage huggingface_hub tqdm
python upload_models_to_gcs.py
```

**GCS bucket layout:**

```
gs://<YOUR_BUCKET_NAME>/
  finetuned_encoder/        ← your fine-tuned SentenceTransformer
                              (or all-MiniLM-L6-v2 if you don't have one)
  all-MiniLM-L6-v2/         ← base encoder fallback (HuggingFace)
  bge-reranker-v2-m3/       ← cross-encoder reranker (HuggingFace)
```

> **No fine-tuned model?** Set `EMBEDDING_MODEL_PATH=/models/all-MiniLM-L6-v2` in your Cloud Run env vars and the service will use the base encoder from HuggingFace. Upload it with:
> ```python
> push_hf_model(bucket, "sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2")
> push_hf_model(bucket, "BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3")
> ```

### Step 4 — Create Service Account for GitHub Actions

```bash
gcloud iam service-accounts create <YOUR_SA_NAME> \
  --display-name="GitHub Actions deployer"

SA="<YOUR_SA_NAME>@<YOUR_PROJECT_ID>.iam.gserviceaccount.com"
PROJECT=<YOUR_PROJECT_ID>
COMPUTE_SA="<PROJECT_NUMBER>-compute@developer.gserviceaccount.com"

# Required roles
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:$SA" --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:$SA" --role="roles/run.developer"

gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:$SA" --role="roles/storage.admin"

# Allow the deployer SA to act as the Cloud Run compute SA
gcloud iam service-accounts add-iam-policy-binding $COMPUTE_SA \
  --member="serviceAccount:$SA" \
  --role="roles/iam.serviceAccountUser"
```

Grant the Cloud Run compute SA read access to the model bucket:

```bash
gcloud storage buckets add-iam-policy-binding gs://<YOUR_BUCKET_NAME> \
  --member="serviceAccount:$COMPUTE_SA" \
  --role="roles/storage.objectViewer"
```

Export the service account key (JSON) — you'll add this to GitHub Secrets:

```bash
gcloud iam service-accounts keys create gcp-sa-key.json \
  --iam-account=$SA
```

### Step 5 — First Manual Deploy

After GitHub Actions is configured (next section), trigger the first deploy manually or push a change to `embedding_service/`. Alternatively, deploy once by hand:

```bash
IMAGE="<YOUR_REGION>-docker.pkg.dev/<YOUR_PROJECT_ID>/<YOUR_REPO_NAME>/embedding-service:latest"

gcloud run deploy <YOUR_SERVICE_NAME> \
  --image=$IMAGE \
  --region=<YOUR_REGION> \
  --gpu=1 --gpu-type=<GPU_TYPE> \
  --cpu=8 --memory=32Gi \
  --max-instances=<N> --concurrency=<C> \
  --timeout=300s \
  --execution-environment=gen2 \
  --port=8001 \
  --cpu-boost \
  --add-volume=name=models,type=cloud-storage,bucket=<YOUR_BUCKET_NAME> \
  --add-volume-mount=volume=models,mount-path=/models \
  --set-env-vars="EMBEDDING_MODEL_PATH=/models/finetuned_encoder,RERANKER_MODEL_NAME=/models/bge-reranker-v2-m3"
```

### Step 6 — Verify the Service

```bash
URL=$(gcloud run services describe <YOUR_SERVICE_NAME> \
  --region=<YOUR_REGION> --format="value(status.url)")

curl "$URL/health"
# Expected: {"status":"ok","device":"cuda","encoder":"...","reranker":"..."}
```

---

## CI/CD — GitHub Actions

The workflow at [`.github/workflows/deploy-embedding-service.yml`](.github/workflows/deploy-embedding-service.yml) automatically builds and deploys the embedding service to Cloud Run whenever files under `embedding_service/` change on `main`.

### Required GitHub Secrets and Variables

Navigate to **Settings → Secrets and variables → Actions** in your repository and add:

| Type | Name | Value |
|---|---|---|
| Secret | `GCP_SA_KEY` | Contents of `gcp-sa-key.json` (the service account key JSON) |
| Variable | `GCP_PROJECT` | Your GCP project ID |
| Variable | `GCS_MODEL_BUCKET` | Your GCS bucket name |

### What the Workflow Does

1. **Authenticates** to GCP using the service account key
2. **Builds** the `embedding_service/` Docker image for `linux/amd64` using Docker BuildKit with layer caching
3. **Pushes** two tags to Artifact Registry: `:latest` and `:<commit-sha>`
4. **Deploys** the commit-SHA-tagged image to Cloud Run with GPU, GCS volume mount, and env vars
5. **Prints** the deployed service URL

```
Trigger: push to main, changes in embedding_service/**
  │
  ├─ docker/build-push-action  →  <YOUR_REGION>-docker.pkg.dev/<YOUR_PROJECT_ID>/<YOUR_REPO_NAME>/embedding-service:<sha>
  │                                                                                  │
  └─ google-github-actions/deploy-cloudrun  ◄────────────────────────────────────────
       GPU=<GPU_TYPE>, CPU=8, Memory=32Gi
       Volume: gs://<bucket> → /models
       Env: EMBEDDING_MODEL_PATH, RERANKER_MODEL_NAME
```

### Artifact Registry Cleanup

To avoid unbounded image accumulation, configure a cleanup policy in the Artifact Registry UI:
- **Keep policy:** retain the 3 most recent versions
- **Delete policy:** delete older versions

Or via CLI:

```bash
gcloud artifacts repositories set-cleanup-policies <YOUR_REPO_NAME> \
  --location=<YOUR_REGION> \
  --policy='[{"name":"keep-recent","action":{"type":"Keep"},"mostRecentVersions":{"keepCount":3}}]'
```

---

## Quickstart

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- GCP embedding service deployed (see [GCP Setup](#gcp-setup--embedding-service)) **or** a local embedding service running on port 8001
- The Cloud Run service URL (or `http://localhost:8001` for local mode)

### Step 1 — Environment variables

Copy the example env file and fill in values:

```bash
cp .env.example dev.env
```

Minimum required values in `dev.env`:

```bash
OPENSEARCH_BM25_INDEX=products_bm25
OPENSEARCH_HNSW_INDEX=products_hnsw

# URL of the embedding service (Cloud Run or local)
EMBEDDING_SERVICE_URL=https://project-name-<hash>.us-west1.run.app
```

### Step 2 — Start local services

```bash
docker compose up --build
```

This starts OpenSearch, the API, and the UI. The API proxies encode/rerank calls to `EMBEDDING_SERVICE_URL`.

| Service | URL |
|---|---|
| Search UI (Streamlit) | http://localhost:8501 |
| API (FastAPI) | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| OpenSearch | http://localhost:9200 |

### Step 3 — Ingest data

Before searching, index products into OpenSearch. Run the ingestion notebook:

```
notebooks/database_ingestion.ipynb
```

This notebook:

1. Loads and curates the ESCI and WANDS datasets using `DatasetMerger`
2. Creates the BM25 index and ingests all products
3. Creates the HNSW index and ingests all product embeddings (calls the embedding service for encoding)

> For local development on a low resource machine, use `merger.build_dev_sample()` instead of `merger.build()` to index a smaller ~5k product subset.

### Step 4 — Search

Open `http://localhost:8501` to use the Streamlit search UI, or call the API directly:

```bash
# BM25 search
curl -X POST http://localhost:8000/search/bm25 \
  -H "Content-Type: application/json" \
  -d '{"query": "red running shoes", "k": 5}'

# Hybrid search with reranking
curl -X POST http://localhost:8000/search/hybrid_rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "red running shoes", "k": 5}'
```

### (Optional) Local Python environment

```bash
bash setup.sh
conda activate product_search_env
```

---

## Repository Structure

```
Product-Search-Engine/
├── .github/
│   └── workflows/
│       └── deploy-embedding-service.yml   # CI/CD: build + deploy to Cloud Run on push
│
├── api/
│   ├── Dockerfile                         # Lightweight API image (no ML models)
│   └── main.py                            # FastAPI: /search/bm25, /hnsw, /hybrid, /hybrid_rerank
│
├── embedding_service/
│   ├── Dockerfile                         # pytorch/cuda base; models loaded from GCS at runtime
│   ├── main.py                            # FastAPI: POST /encode, POST /rerank, GET /health
│   └── requirements.txt
│
├── notebooks/
│   └── database_ingestion.ipynb           # End-to-end: curate → index → verify
│
├── src/product_search/
│   ├── data_curation.py                   # ESCIProcessor, WANDSProcessor, DatasetMerger
│   ├── search_pipeline.py                 # OpenSearchInference: BM25, HNSW, Hybrid, Rerank
│   ├── benchmarking.py                    # End-to-end evaluation: all strategies + agentic
│   └── agent/
│       ├── state.py                       # AgentState TypedDict
│       ├── nodes.py                       # understand_query, retrieve, assess_results, decompose nodes
│       ├── graph.py                       # LangGraph StateGraph + run_agent() public entry point
│       ├── llm.py                         # ChatGoogleGenerativeAI setup (Vertex AI, global endpoint)
│       └── prompts.toml                   # System + user prompt templates for all LLM nodes
│
├── ui/
│   ├── Dockerfile
│   └── app.py                             # Streamlit search UI
│
├── upload_models_to_gcs.py                # One-time script: upload HF/local models to GCS
├── docker-compose.yml                     # OpenSearch + API + UI (embedding service is remote)
├── .env.example                           # Template for dev.env
├── pyproject.toml
└── requirements.txt
```

---

> **Summary:** Hybrid retrieval with a fine-tuned bi-encoder + cross-encoder reranking (candidate pool C=25) is the top-performing configuration at all depth cutoffs (NDCG@5 = 0.682, NDCG@10 = 0.661, NDCG@20 = 0.651). BM25 uses multi-field boosting across `title`, `brand_text`, `bullets`, and `description`; the dense encoder uses a short `encode_text` field (title + brand + color) matching its fine-tuning representation. Encoding and reranking are handled by a GPU-backed Cloud Run service that loads models from GCS at boot. An experimental **agentic search layer** (LangGraph + Gemini Flash) wraps hybrid_rerank with query understanding and result assessment; v1 benchmarks at NDCG@20 = 0.628 (−3.5% vs baseline) due to marginal noise from query rewriting on well-specified queries — a v2 design using structured filter extraction and query-type routing is planned.