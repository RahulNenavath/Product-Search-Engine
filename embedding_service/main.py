"""
embedding_service/main.py

Lightweight FastAPI inference service for embedding and reranking.
Runs on Cloud Run GPU (NVIDIA L4). Has no knowledge of OpenSearch or search logic.

Endpoints:
  GET  /health  — liveness check, reports device and model info
  POST /encode  — encode a batch of texts into embedding vectors
  POST /rerank  — score (query, text) pairs with a cross-encoder

Environment variables:
  EMBEDDING_MODEL_PATH   path to fine-tuned SentenceTransformer (mounted from GCS)
  RERANKER_MODEL_NAME    HuggingFace model ID for the cross-encoder reranker
"""

import os
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/models/finetuned_encoder")
RERANKER_MODEL_NAME  = os.getenv("RERANKER_MODEL_NAME",  "/models/bge-reranker-v2-m3")

# ── Boot ──────────────────────────────────────────────────────────────────────

print("torch:", torch.__version__)
print("[BOOT] cuda available:", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[BOOT] Running on device: {device}")

print(f"[BOOT] Loading encoder from '{EMBEDDING_MODEL_PATH}'...")
encoder: SentenceTransformer = SentenceTransformer(EMBEDDING_MODEL_PATH, device=device)

print(f"[BOOT] Loading reranker '{RERANKER_MODEL_NAME}'...")
reranker: CrossEncoder = CrossEncoder(RERANKER_MODEL_NAME, device=device)

print("[BOOT] Ready.")

# ── Schemas ───────────────────────────────────────────────────────────────────

class EncodeRequest(BaseModel):
    texts: List[str]
    normalize: bool = True


class EncodeResponse(BaseModel):
    embeddings: List[List[float]]


class RerankRequest(BaseModel):
    query: str
    texts: List[str]


class RerankResponse(BaseModel):
    scores: List[float]


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Embedding Service", version="0.1.0")


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": device,
        "encoder": EMBEDDING_MODEL_PATH,
        "reranker": RERANKER_MODEL_NAME,
    }


@app.post("/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest) -> EncodeResponse:
    if not req.texts:
        raise HTTPException(status_code=422, detail="texts must be non-empty")
    vecs = encoder.encode(
        req.texts,
        normalize_embeddings=req.normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return EncodeResponse(embeddings=vecs.tolist())


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    if not req.texts:
        raise HTTPException(status_code=422, detail="texts must be non-empty")
    pairs = [(req.query, text) for text in req.texts]
    scores = reranker.predict(pairs, show_progress_bar=False)
    return RerankResponse(scores=[float(s) for s in scores])
