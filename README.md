# Product Search Engine

A modular **product search engine** built to experiment with and benchmark **retrieval + ranking** strategies for e-commerce search. The goal is to identify configurations that return **highly relevant products** for natural-language queries, balancing **precision, recall, and latency**.

The project benchmarks methods using publicly available product-search datasets:
- **Amazon ESCI** 
- **Wayfair WANDS**

---

## Problem Statement

In e-commerce, search quality directly impacts discovery and conversion. Given a user query (often short, ambiguous, and noisy), the system must retrieve a small set of candidate products and rank them so that the most relevant items appear at the top.

This repository implements and compares multiple retrieval approaches:

1. **BM25** (lexical retrieval)  
   - Strong baseline for exact term matches, brand names, and structured tokens.

2. **Dense retrieval** using **HNSW k-NN** over learned embeddings (semantic retrieval)  
   - Improves recall for paraphrases and intent-level matches beyond keyword overlap.

3. **Hybrid retrieval** (BM25 + HNSW) with rank aggregation and optional reranking  
   - **Reciprocal Rank Fusion (RRF)** to combine lexical + semantic candidates.
   - Optional **cross-encoder reranking** to refine the final ordering over a candidate pool.

---

## Data Description

This project benchmarks retrieval and ranking methods using two public product-search relevance datasets: **Wayfair WANDS** and **Amazon Shopping Queries (ESCI)**. Both datasets provide (query, product) pairs with human relevance judgements, enabling reproducible offline evaluation of search quality.

### Wayfair WANDS (Wayfair ANnotation Dataset)

**WANDS** is a curated product search relevance dataset designed for objective benchmarking in e-commerce search. It contains **42,994 candidate products**, **480 queries**, and **233,448 (query, product) relevance judgements**.  [WANDs-GitHub-Repo](https://github.com/wayfair/WANDS)

**Files and schema (CSV):**
- `product.csv` (catalog / candidates)
  - `product_id`, `product_name`, `product_class`, `category_hierarchy` (delimited by `/`)
  - `product_description`, `product_features` (`|`-delimited attribute:value pairs)
  - `rating_count`, `average_rating`, `review_count`
- `query.csv`
  - `query_id`, `query`, `query_class`
- `label.csv` (annotations)
  - `id`, `query_id`, `product_id`, `label` ∈ {`Exact`, `Partial`, `Irrelevant`}

### Amazon Shopping Queries Dataset (ESCI)

Amazon’s **Shopping Queries Dataset** (commonly referred to as **ESCI**) is a large-scale benchmark for semantic matching between user shopping queries and products. For each query, it provides up to **40** candidate results with a graded relevance label from the **E/S/C/I** scheme.  [Amazon-ESCI-GitHub](https://github.com/amazon-science/esci-data)

**Relevance labels (ESCI):**
- **Exact (E):** satisfies all query specifications
- **Substitute (S):** partially satisfies the query but can function as a substitute
- **Complement (C):** not a match itself, but can be used with an exact match
- **Irrelevant (I):** unrelated or misses a central requirement  [Amazon-ESCI-Paper-Link](https://ar5iv.org/pdf/2206.06588)

**Dataset size and versions:**
- Two official versions are provided:
  - **Small (Task 1 / ranking):** ~48,300 unique queries and ~1.1M judgements
  - **Large (Tasks 2–3):** ~130,652 unique queries and ~2.6M judgements
- Multilingual queries are included (English, Japanese, Spanish).

**Example-level fields (used as features / metadata):**  
Each row corresponds to a `<query, product>` judgement and includes identifiers, locale, label, and product text fields such as title, description, bullet points, brand, and color.

### Unified Product Representation in This Repository

To support both **lexical retrieval (BM25)** and **dense retrieval (HNSW k-NN)** in a consistent way, products from both datasets are normalized into a single OpenSearch document schema:

- `product_id` (keyword): stable identifier for retrieval and evaluation
- `source` (keyword): dataset origin (ESCI vs WANDS)
- `full_text` (text): a compact searchable text built from common product fields
- `metadata` (object): full raw product attributes preserved for debugging and analysis
- `embedding` (knn_vector, HNSW only): dense vector representation of `full_text`

**Text construction (`full_text`):**  
For cross-dataset compatibility, the default document text concatenates the most consistently available high-signal fields:
- ESCI: `product_title`, `product_brand`
- WANDS: `product_name`, `product_class`

This yields a concise representation suitable for BM25 indexing and embedding generation, while richer attributes (e.g., descriptions, features, bullets) remain available under `metadata` for future expansion of the indexed text.

## Search Engine Architecture

This repository follows a **modular retrieval → fusion/rerank → evaluation** methodology to compare multiple search configurations under a consistent interface. The architecture is intentionally split into **two OpenSearch indices** (lexical and vector) so that each retrieval paradigm can be tuned independently while sharing the same document schema and metadata.

### 1) Core Components

**A. Document normalization (shared across datasets)**  
During ingestion, each product (from ESCI or WANDS) is normalized into a single document structure:

- `product_id` (keyword)
- `source` (keyword): dataset identifier (e.g., `ESCI`, `WANDS`)
- `full_text` (text): compact searchable text representation built from high-signal fields
- `metadata` (object): full product attributes preserved for analysis/debugging
- `embedding` (knn_vector): only in the vector index

`full_text` is built using a lightweight cross-dataset heuristic (title/name + brand/class when available). This keeps the baseline comparable across BM25 and dense retrieval while retaining richer fields in `metadata` for later iterations.

---

**B. Index layer (OpenSearch)**

1) **BM25 index (lexical retrieval)**  
A standard text index optimized for keyword matching:

- `full_text`: `text`
- optional boosted fields inside `metadata` (e.g., product title/name)

2) **HNSW index (semantic retrieval)**  
A k-NN enabled index storing dense embeddings:

- `embedding`: `knn_vector` with `hnsw` method  
- ANN parameters (tunable): `m`, `ef_construction`, `ef_search`
- Engine configured for vector search (e.g., FAISS)

This separation allows:
- independent tuning of BM25 analyzers vs ANN graph parameters,
- simpler ablation studies (lexical vs semantic vs hybrid),
- consistent filtering (`source`) and metadata inspection across strategies.

---

### 2) Online Retrieval Methodology (Inference)

All retrieval strategies are exposed through a single inference client (`OpenSearchInference`) and share a common request/response contract:

- **Input:** `query`, `k`, optional `filter_source`
- **Output:** ranked `SearchHit` list with `product_id`, `score`, and optional `full_text/metadata`

#### Strategy A — BM25 Retrieval (Lexical)
**Query plan**
- Uses a `multi_match` query over:
  - `metadata.product_title^4`
  - `metadata.product_name^4`
  - `full_text`
- Optional filtering by dataset source

---

#### Strategy B — HNSW k-NN Retrieval (Dense / Semantic)
**Query plan**
1) Encode the user query into a dense vector using a `SentenceTransformer`.
2) Issue a `knn` query against `embedding` with top-*k* ANN retrieval.
3) Optionally apply `source` filter via a boolean query.

---

#### Strategy C — Hybrid Retrieval via Reciprocal Rank Fusion (RRF)
**Goal:** combine lexical + semantic signals without learning-to-rank.

The hyper-params like `C` is set to 30, and `K` is set to 10.

**Pipeline**
1) Retrieve a **candidate pool** of size `C` from BM25.
2) Retrieve a **candidate pool** of size `C` from HNSW.
3) Fuse ranked lists with **Reciprocal Rank Fusion**:
4) Return top-*k* fused documents.

---

#### Strategy D — Hybrid Retrieval + Cross-Encoder Reranking
**Goal:** maximize ranking quality using a stronger (slower) model over a small candidate set.

**Pipeline**
1) Retrieve top `C` from BM25 and top `C` from HNSW.
2) **Union + deduplicate** candidates by `product_id`.
3) Build (query, document_text) pairs where `document_text = full_text`.
4) Score pairs using a **CrossEncoder** and sort by reranker score.
5) Return top-*k* reranked documents.

---

## Encoder Fine-tuning (Dense Retrieval)

Dense retrieval quality depends heavily on how well the encoder maps **(query, product)** text into a shared embedding space. While an off-the-shelf encoder can provide a reasonable semantic baseline, it is typically **not optimized for the specific relevance notion** in a product-search benchmark (e.g., exact vs substitute vs complement). To address this, this project fine-tunes a SentenceTransformer encoder on the training split using weakly supervised pairs derived from dataset relevance labels.

### Training Objective

The fine-tuning pipeline trains a bi-encoder such that:

- the **query embedding** is close to embeddings of **relevant products** (positives),
- and far from **non-relevant products** (negatives), including *hard* in-batch negatives.

This improves semantic retrieval (HNSW k-NN) by aligning the embedding geometry with the dataset’s relevance judgments.

### Data Construction (Query → Positive Product)

Training data is constructed from three inputs:

- `qrels`: relevance judgements, stored as `{query_id: {product_id: gain}}`
- `query_table`: a table mapping `query_id → query`
- `product_store`: a dictionary mapping `product_id → product_metadata`

**Pair generation strategy (current):**
- For each query, select **one positive product**: the product with the **highest gain**.
- Convert the product metadata into a compact text string using `build_full_text()`:
  - ESCI-like fields: `product_title`, `product_brand`
  - WANDS-like fields: `product_name`, `product_class`
- Emit a training pair:  
  **(anchor = query, positive = product_full_text)**

### Loss Function: CachedGIST (with Guide Model)

The training uses **`CachedGISTEmbedLoss`**, which is designed for efficient contrastive learning with **in-batch negatives** while reducing the impact of **false negatives**.

- **Trainable model (`model`)**: the base encoder being fine-tuned. We select `all-MiniLM-L6-v2` model as the base model to be fine-tuned.
- **Guide/teacher model (`guide`)**: a stronger encoder (the `BAAI/bge-m3` encoder model) used to identify/filter likely false negatives during training (e.g., a high-quality retrieval model such as BGE-M3).
- **Effect:** improves hard-negative training quality without requiring explicit negative mining pipelines.

### Training Configuration

The script is designed for an **NVIDIA GPU machine** (CUDA required) and uses:
- base model = `all-MiniLM-L6-v2`, and guide model = `BAAI/bge-m3`.
- `SentenceTransformerTrainer` + `SentenceTransformerTrainingArguments`
- Mixed precision training (`fp16=True`) for throughput
- Tunable hyperparameters:
  - `epochs`
  - `learning_rate`
  - `warmup_ratio`
  - `weight_decay`
  - `per_device_train_batch_size`

---

## Evaluation and Metrics

This project evaluates retrieval and ranking quality on held-out **test sets** using standard information-retrieval metrics computed at fixed cutoffs **K ∈ {5, 10, 20}**. These metrics measure how well each configuration surfaces relevant products near the top of the ranked list—where users actually look.

### Metrics Used

#### 1) Recall@K
**What it measures:** Coverage of relevant items within the top-K results.

- **Definition:**
  Recall@K is the fraction of all relevant items for a query that appear within the top K retrieved results.  
  Concretely: **(number of relevant products in the top K results) divided by (total number of relevant products for that query)**.  
  The final Recall@K is typically reported as the average across all queries.

- **Interpretation:** Higher Recall@K means the system is better at *retrieving* relevant candidates.  
- **Why it matters:** Particularly important for two-stage systems: a strong first-stage retriever should maximize recall so that rerankers have good candidates to work with.

#### 2) NDCG@K (Normalized Discounted Cumulative Gain)
**What it measures:** Ranking quality with **position discounting** and **graded relevance**.

- **Key idea:** Putting a highly relevant item at rank 1 is better than rank 10, and relevance can be graded (not just relevant/irrelevant).
- **Definition (high level):**
  - DCG@K rewards relevant items near the top with logarithmic discounting:
    DCG@K is computed by summing a gain for each result position from rank 1 to K, where:
    (1) more relevant items contribute higher gain, and  
    (2) contributions are **discounted** as rank increases (items appearing later contribute less than those at the top).  
    In practice, each item’s contribution is divided by a log-based factor depending on its rank.
  - NDCG@K normalizes DCG by the best possible ranking (IDCG):
    NDCG@K is computed as **DCG@K divided by IDCG@K**, where IDCG@K is the DCG@K of an ideal ranking (all most relevant items placed at the top).  
    This normalization makes scores comparable across queries and bounded between 0 and 1.

- **Interpretation:** Higher NDCG@K means **better ordering** of results, especially at the top ranks.
- **Why it matters:** This is typically the most informative metric for user-facing search because it emphasizes the head of the ranking.

#### 3) MRR@K (Mean Reciprocal Rank)
**What it measures:** How quickly the first relevant result appears.

- **Definition:**
  For each query, find the rank position (1..K) of the **first relevant** product in the returned list.  
  The reciprocal rank for that query is **1 divided by that rank** (e.g., rank 1 → 1.0, rank 2 → 0.5, rank 5 → 0.2).  
  If no relevant product appears in the top K, the reciprocal rank is 0.  
  MRR@K is the average reciprocal rank across all queries.

- **Interpretation:** Higher MRR indicates users are more likely to find a relevant product immediately.
- **Why it matters:** Strong proxy for “time-to-success” in navigational or single-intent queries.

---

### Evaluation Cutoffs (K)

All metrics are computed at:

- **K = 5** (very top results / most user impact)
- **K = 10** (typical first-page ranking)
- **K = 20** (deeper retrieval quality and robustness)

---

### Test-Set Results (NDCG@K)

| Configuration              | NDCG@5 | NDCG@10 | NDCG@20 |
|---------------------------|:------:|:-------:|:-------:|
| **BM25**                  |  0.37  |  0.33   |  0.32   |
| **Base Encoder (HNSW)**   |  0.32  |  0.29   |  0.29   |
| **Fine-tuned Encoder (HNSW)** | 0.37 | 0.34 | 0.33 |
| **Hybrid Fine-tuned Encoder + RRF**          |  0.40  |  0.36   |  0.36   |
| **Hybrid Fine-tuned Encoder + Cross-Encoder**| **0.47** | **0.42** | **0.41** |

---

### Findings and Interpretation

1) **Fine-tuning improves dense retrieval materially.**  
The fine-tuned encoder HNSW matches or slightly exceeds BM25 across all cutoffs (e.g., **0.37 vs 0.37** at NDCG@5; **0.34 vs 0.33** at NDCG@10). This indicates the fine-tuned embedding model better captures domain relevance than the base embedding model.

2) **Hybrid with Fine-tuned Encoder retrieval (RRF) shows strong improvement.**  
Hybrid + RRF aggregation improves over either single retriever, reaching **0.40 / 0.36 / 0.36** at K={5,10,20}. This suggests BM25 and dense retrieval retrieve complementary relevant items, and simple fusion improves overall ranking quality without heavy modeling.

3) **Hybrid Fine-tuned Encoder + Cross-Encoder is the best-performing configuration.**  
The cross-encoder reranked hybrid system achieves the highest NDCG at every cutoff (**0.47 / 0.42 / 0.41**). This is consistent with a two-stage architecture where:
- stage 1 maximizes candidate recall (BM25 + HNSW),
- stage 2 performs high-precision ranking with richer interaction modeling (cross-encoder).

---

> In short: **Hybrid + Cross-Encoder** is the top-performing search configuration on the test sets by NDCG@{5,10,20}, followed by **Hybrid + RRF**.

---

## Final Architecture

A production-ready version of this project converges on a **hybrid retrieval + reranking** design:

### Recommended production flow

1. **Query understanding**
   - User query comes into API.

2. **Candidate generation (Hybrid)**
   - Run **BM25** (high precision for exact matches, brands, SKUs)
   - Run **HNSW dense retrieval** (semantic recall)
   - Candidates from both search engines are appeneded into a list.

3. **Reranking**
   - The top-C candidates from list, are reranked using **cross-encoder**.
   - Return top-K results.

4. **Metric**
   - Offline evaluation with qrels (Recall/MRR/NDCG) metrics.

### Components

- **OpenSearch**
  - BM25 index (`products_bm25`)
  - HNSW index (`products_hnsw`)
- **Embedding model**
  - Fine tuned model: `all-MiniLM-L6-v2`
  - Reranker model: `BAAI/bge-reranker-v2-m3`
- **API**
  - FastAPI service for search endpoints

---

## Quickstart (Development)

### 1) Environment variables

Copy `.env.example` into a `dev.env` file (Docker Compose expects `dev.env`):

- `OPENSEARCH_HOST`, `OPENSEARCH_PORT`
- `OPENSEARCH_BM25_INDEX`, `OPENSEARCH_HNSW_INDEX`

### 2) Run OpenSearch + API

```bash
docker compose up --build
```

API should be available at:
- http://localhost:8000
- Health: `GET /health`
- BM25: `POST /search/bm25`
- HNSW: `POST /search/hnsw`

### 3) (Optional) Local Python environment

```bash
bash setup.sh
conda activate product_search_env
```

---

## Repository Structure

- `api/` — FastAPI service for search
- `src/product_search/`
  - `data_ingestion.py` — build indices + ingest docs/embeddings
  - `search_pipeline.py` — BM25/HNSW/Hybrid inference utilities
  - `evaluation.py` — ranx-based metrics evaluation
  - `benchmarking.py` — batch benchmarking script
  - `finetune_encoder.py` — SentenceTransformer fine-tuning script
- `notebooks/` — exploration and pipeline notebooks (data curation, ingestion, fine-tuning, benchmarking)
- `docker-compose.yml` — OpenSearch + API orchestration
