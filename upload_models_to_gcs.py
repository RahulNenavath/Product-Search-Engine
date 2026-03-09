"""
One-time script to upload all model weights to GCS.
Cloud Run mounts the bucket at /models

Requirements (install once, not part of main project deps):
  pip install google-cloud-storage huggingface_hub tqdm

Usage:
  # Uses defaults from environment or falls back to hardcoded values below
  python upload_models_to_gcs.py

  # Override bucket or project
  GCS_MODEL_BUCKET=my-bucket GCP_PROJECT=my-project python scripts/upload_models_to_gcs.py

Authentication:
  Requires Application Default Credentials:
    gcloud auth application-default login
"""

import os
import shutil
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage
from huggingface_hub import snapshot_download
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv("dev.env")

PROJECT    = os.getenv("GCP_PROJECT",       "product-search-489623")
BUCKET     = os.getenv("GCS_MODEL_BUCKET",  "product-search-models")

# Models to upload — edit this list to add/remove models
MODELS = [
    {
        "type":       "local",
        "source":     "src/product_search/finetuned_encoder",
        "gcs_prefix": "finetuned_encoder",
        "label":      "Fine-tuned encoder (local)",
    },
    {
        "type":       "huggingface",
        "repo_id":    "sentence-transformers/all-MiniLM-L6-v2",
        "gcs_prefix": "all-MiniLM-L6-v2",
        "label":      "Base encoder (HuggingFace)",
    },
    {
        "type":       "huggingface",
        "repo_id":    "BAAI/bge-reranker-v2-m3",
        "gcs_prefix": "bge-reranker-v2-m3",
        "label":      "Cross-encoder reranker (HuggingFace)",
    },
]

# ── GCS helpers ───────────────────────────────────────────────────────────────

def _get_existing_blobs(bucket_obj: storage.Bucket, prefix: str) -> dict[str, int]:
    """Return {blob_name: size_bytes} for all blobs under prefix."""
    return {
        blob.name: blob.size
        for blob in bucket_obj.list_blobs(prefix=prefix)
    }


def _upload_directory(
    bucket_obj: storage.Bucket,
    local_dir: Path,
    gcs_prefix: str,
) -> tuple[int, int]:
    """
    Upload all files from local_dir to gs://<bucket>/<gcs_prefix>/.
    Skips files that already exist in GCS with the same size (rsync-style).

    Returns (uploaded_count, skipped_count).
    """
    existing = _get_existing_blobs(bucket_obj, gcs_prefix)

    files = [f for f in local_dir.rglob("*") if f.is_file()]
    uploaded = skipped = 0

    for local_file in tqdm(files, desc=f"  Uploading to {gcs_prefix}/", unit="file"):
        relative   = local_file.relative_to(local_dir)
        blob_name  = f"{gcs_prefix}/{relative}".replace("\\", "/")
        local_size = local_file.stat().st_size

        # Skip if already in GCS with matching size
        if blob_name in existing and existing[blob_name] == local_size:
            skipped += 1
            continue

        blob = bucket_obj.blob(blob_name)
        blob.upload_from_filename(str(local_file))
        uploaded += 1

    return uploaded, skipped


# ── Public functions ──────────────────────────────────────────────────────────

def push_local_model(
    bucket_obj: storage.Bucket,
    local_path: str | Path,
    gcs_prefix: str,
) -> None:
    """
    Upload a local model folder to gs://<bucket>/<gcs_prefix>/.
    Skips files already present in GCS with matching size.

    Args:
        bucket_obj: google.cloud.storage.Bucket instance
        local_path: path to local model directory (e.g. src/product_search/finetuned_encoder)
        gcs_prefix: destination prefix inside the bucket (e.g. "finetuned_encoder")
    """
    local_dir = Path(local_path).resolve()

    if not local_dir.is_dir():
        raise FileNotFoundError(
            f"Local model directory not found: {local_dir}\n"
            "Run this script from the project root."
        )

    print(f"  Source : {local_dir}")
    print(f"  Dest   : gs://{bucket_obj.name}/{gcs_prefix}/")

    uploaded, skipped = _upload_directory(bucket_obj, local_dir, gcs_prefix)
    print(f"  Done   : {uploaded} uploaded, {skipped} skipped (already in GCS)")


def push_hf_model(
    bucket_obj: storage.Bucket,
    repo_id: str,
    gcs_prefix: str,
) -> None:
    """
    Download a HuggingFace model to a temp directory, upload to GCS, then delete locally.
    Cleans up the temp directory even if the upload fails.

    Args:
        bucket_obj: google.cloud.storage.Bucket instance
        repo_id:    HuggingFace repo ID (e.g. "BAAI/bge-reranker-v2-m3")
        gcs_prefix: destination prefix inside the bucket (e.g. "bge-reranker-v2-m3")
    """
    print(f"  Source : HuggingFace — {repo_id}")
    print(f"  Dest   : gs://{bucket_obj.name}/{gcs_prefix}/")

    # Check if already fully uploaded by comparing blob count
    existing = _get_existing_blobs(bucket_obj, gcs_prefix)
    if existing:
        print(f"  Found {len(existing)} existing blobs — will skip unchanged files.")

    tmp_dir = tempfile.mkdtemp(prefix="hf_model_")
    try:
        print(f"  Downloading {repo_id} to temp dir...")
        local_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                local_dir=tmp_dir,
                local_dir_use_symlinks=False,   # ensure real files, not symlinks
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
        )

        uploaded, skipped = _upload_directory(bucket_obj, local_dir, gcs_prefix)
        print(f"  Done   : {uploaded} uploaded, {skipped} skipped (already in GCS)")

    finally:
        print(f"  Cleaning up temp dir {tmp_dir}...")
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    print("=" * 60)
    print("  GCS Model Upload")
    print(f"  Project : {PROJECT}")
    print(f"  Bucket  : gs://{BUCKET}")
    print("=" * 60)
    print()

    # Connect to GCS
    client     = storage.Client(project=PROJECT)
    bucket_obj = client.bucket(BUCKET)

    # Create bucket if it doesn't exist
    if not bucket_obj.exists():
        print(f"[SETUP] Creating bucket gs://{BUCKET}...")
        bucket_obj = client.create_bucket(
            BUCKET,
            location=os.getenv("GCP_REGION", "us-east4"),
        )
        bucket_obj.iam_configuration.uniform_bucket_level_access_enabled = True
        bucket_obj.patch()
        print(f"[SETUP] Bucket created.\n")
    else:
        print(f"[SETUP] Bucket already exists — skipping creation.\n")

    # Upload each model
    for i, model in enumerate(MODELS, 1):
        print(f"[{i}/{len(MODELS)}] {model['label']}")

        if model["type"] == "local":
            push_local_model(
                bucket_obj,
                local_path=model["source"],
                gcs_prefix=model["gcs_prefix"],
            )

        elif model["type"] == "huggingface":
            push_hf_model(
                bucket_obj,
                repo_id=model["repo_id"],
                gcs_prefix=model["gcs_prefix"],
            )

    print()
    print("=" * 60)
    print("  Upload complete. GCS layout:")
    for model in MODELS:
        print(f"    gs://{BUCKET}/{model['gcs_prefix']}/")
    print()
    print("  Next steps:")
    print(f"    1. Add GCS_MODEL_BUCKET={BUCKET} to GitHub repo variables")
    print( "    2. Push a commit to embedding_service/ to trigger first deploy")
    print()
    print("  Cloud Run volume mount (in cloudbuild.yaml deploy step):")
    print(f"    --add-volume=name=model-store,type=cloud-storage,bucket={BUCKET}")
    print( "    --add-volume-mount=volume=model-store,mount-path=/models")
    print("=" * 60)


if __name__ == "__main__":
    main()