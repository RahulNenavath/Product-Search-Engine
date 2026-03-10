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
7. [Evaluation and Metrics](#evaluation-and-metrics)
8. [Results](#results)
9. [System Components and Docker](#system-components-and-docker)
10. [GCP Setup — Embedding Service](#gcp-setup--embedding-service)
11. [CI/CD — GitHub Actions](#cicd--github-actions)
12. [Quickstart](#quickstart)
13. [Repository Structure](#repository-structure)

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

Encoding and reranking are **not performed locally** — all calls are proxied to the remote `embedding_service` running on Cloud Run (see [GCP Setup](#gcp-setup--embedding-service)).

### Strategy A — BM25 (Lexical)

```
query → match on full_text → top-k hits (sorted by BM25 score)
```

### Strategy B — HNSW k-NN (Dense / Semantic)

```
query → POST /encode (embedding_service) → knn on embedding → top-k hits
```

### Strategy C — Hybrid via Reciprocal Rank Fusion (RRF)

```
query → BM25 top-C  ─┐
                      ├→ RRF fusion → top-k
query → HNSW top-C  ─┘
```

Default hyperparameters: `C = 20`, `rrf_k = 60`.

### Strategy D — Hybrid + Cross-Encoder Reranking

```
query → BM25 top-C  ─┐
                      ├→ union/dedup → POST /rerank (embedding_service) → top-k
query → HNSW top-C  ─┘
```

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

### NDCG@K on Test Set (Combined ESCI + WANDS)

| Configuration | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|
| BM25 | 0.5856 | 0.5646 | 0.5484 |
| HNSW (Fine-tuned Encoder) | 0.6017 | 0.5854 | 0.5810 |
| Hybrid RRF (BM25 + HNSW Fine-tuned) | 0.6354 | 0.6175 | **0.6037** |
| **Hybrid + Cross-Encoder Reranker** | **0.6883** | **0.6453** | 0.5588 |

### Key Findings

1. **Fine-tuned encoder outperforms BM25 at every cutoff**, confirming the encoder better captures domain-level relevance beyond keyword overlap.
2. **Hybrid RRF is a strong, calibration-free improvement** — it reaches NDCG@5 of 0.635 without any additional learned components, and leads all methods at NDCG@20 (0.604).
3. **Hybrid + Cross-Encoder is the best configuration at shallow cutoffs** (NDCG@5 = 0.688, +17.6% vs BM25), confirming the value of a two-stage architecture: maximise recall in stage 1, maximise precision in stage 2.
4. **Rank inversion at NDCG@20:** the reranker's benefit is bounded by the size of its candidate pool — Hybrid RRF overtakes it at depth 20 (0.604 vs 0.559), as ordering beyond the reranked window reverts to fused scores.

> **For production ranking** (k ≤ 10): use Hybrid + Cross-Encoder Reranker.
> **For recall-oriented retrieval** (broad candidates, offline indexing): use Hybrid RRF.

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
- Features: search mode toggle (BM25 / HNSW / Hybrid / Hybrid+Rerank), `k` slider, source filter, live API health check, results as expandable cards

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
│   └── ...
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

> **Summary:** Hybrid retrieval with a fine-tuned bi-encoder + cross-encoder reranking is the top-performing configuration (NDCG@5 = 0.688, +17.6% vs BM25). Encoding and reranking are handled by a GPU-backed Cloud Run service that loads models from GCS at boot — keeping Docker images small and model updates independent of application deploys. The local stack (OpenSearch + FastAPI + Streamlit) runs entirely in Docker Compose.