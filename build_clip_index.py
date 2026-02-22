#!/usr/bin/env python3
"""
Build a CLIP-based vector index of all images in iCloudPhotos and iCloudPhotos1,
persisted into a local LanceDB vector database.

Model: openai/clip-vit-base-patch32 (512-dim embeddings)
"""

import sys
import time
import tomllib
from pathlib import Path

import lancedb
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor
import pyarrow as pa

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "config.toml", "rb") as _f:
    _cfg = tomllib.load(_f)

DATA_DIRS        = [BASE_DIR / p for p in _cfg["index"]["data_dirs"]]
DB_PATH          = BASE_DIR / _cfg["index"]["db_path"]
TABLE_NAME       = _cfg["index"]["table_name"]
IMAGE_EXTENSIONS = set(_cfg["index"]["image_extensions"])
MODEL_ID         = _cfg["model"]["model_id"]
EMBED_DIM        = _cfg["model"]["embed_dim"]
BATCH_SIZE       = _cfg["model"]["batch_size"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def collect_images(dirs: list[Path]) -> list[Path]:
    paths = []
    for d in dirs:
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(p)
    print(f"Found {len(paths)} images across {len(dirs)} directories.")
    return paths


def load_model(device: str):
    print(f"Loading CLIP model '{MODEL_ID}' on {device} …")
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    print("Model loaded.")
    return processor, model


@torch.inference_mode()
def embed_images(
    paths: list[Path],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """Return list of {path, embedding, source_dir, filename} dicts."""
    records = []
    total = len(paths)
    failed = 0

    for i in range(0, total, batch_size):
        batch_paths = paths[i : i + batch_size]
        imgs, valid_paths = [], []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(img)
                valid_paths.append(p)
            except (UnidentifiedImageError, OSError, Exception) as e:
                print(f"  [SKIP] {p.name}: {e}")
                failed += 1

        if not imgs:
            continue

        inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
        outputs = model.get_image_features(**inputs)
        # In transformers 5.x get_image_features may return an output object
        features = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
        # L2-normalise so cosine similarity == dot product
        features = features / features.norm(dim=-1, keepdim=True)
        embeddings = features.cpu().float().numpy()

        for p, emb in zip(valid_paths, embeddings):
            records.append({
                "path": str(p),
                "filename": p.name,
                "source_dir": p.parent.name,
                "vector": emb.tolist(),
            })

        done = min(i + batch_size, total)
        pct = done / total * 100
        print(f"  {done}/{total} ({pct:.1f}%)  –  batch {i // batch_size + 1}", end="\r")

    print()  # newline after \r progress
    print(f"Embedded {len(records)} images  ({failed} skipped).")
    return records


def build_db(records: list[dict]) -> None:
    print(f"Writing {len(records)} records to LanceDB at '{DB_PATH}' …")

    schema = pa.schema([
        pa.field("path",       pa.utf8()),
        pa.field("filename",   pa.utf8()),
        pa.field("source_dir", pa.utf8()),
        pa.field("vector",     pa.list_(pa.float32(), EMBED_DIM)),
    ])

    db = lancedb.connect(str(DB_PATH))

    # Drop existing table so re-runs start fresh
    if TABLE_NAME in db.list_tables().tables:
        db.drop_table(TABLE_NAME)
        print(f"  Dropped existing '{TABLE_NAME}' table.")

    table = db.create_table(TABLE_NAME, schema=schema)
    table.add(records)

    # Create an ANN index for fast similarity search
    print("  Building ANN index …")
    table.create_index(
        metric="cosine",
        vector_column_name="vector",
        num_partitions=64,
        num_sub_vectors=32,
    )

    print(f"Done. Table '{TABLE_NAME}' has {table.count_rows()} rows.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    t0 = time.time()

    image_paths = collect_images(DATA_DIRS)
    if not image_paths:
        print("No images found. Exiting.")
        sys.exit(1)

    processor, model = load_model(device)
    records = embed_images(image_paths, processor, model, device)
    build_db(records)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s  ({elapsed / len(image_paths) * 1000:.1f} ms/image)")


if __name__ == "__main__":
    main()
