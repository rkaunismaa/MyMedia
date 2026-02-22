# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Python venv:** `.mymedia/` — always invoke as `.mymedia/bin/python`
- **No pip in venv** — use `uv pip install <pkg> --python .mymedia/bin/python`
- **GPU:** NVIDIA GeForce RTX 4090 with CUDA 12.6
- **Key packages:** torch 2.10.0+cu126, lancedb 0.29.2, transformers 5.2.0, fastapi, uvicorn, pyarrow (no pandas installed)

## Commands

```bash
# Rebuild the CLIP vector index from scratch (~2 min on RTX 4090)
TRANSFORMERS_VERBOSITY=error .mymedia/bin/python build_clip_index.py

# Start the search web app (http://localhost:8000)
TRANSFORMERS_VERBOSITY=error .mymedia/bin/python app.py
```

## Project structure

| Path | Purpose |
|---|---|
| `build_clip_index.py` | Scans image dirs, runs CLIP inference, writes LanceDB index |
| `app.py` | FastAPI web app — text search UI + image serving on port 8000 |
| `config.toml` | All tuneable parameters (dirs, model ID, embed dim, batch size) |
| `data/iCloudPhotos/` | First iCloud photo export |
| `data/iCloudPhotos1/` | Second iCloud photo export |
| `clip_index.lancedb/` | Persisted LanceDB vector database (gitignored) |

## config.toml

Changing `model_id` or `embed_dim` requires a full index rebuild. Current model is `openai/clip-vit-large-patch14` (768-dim). Both `build_clip_index.py` and `app.py` read this file at startup.

## Web app (`app.py`)

- Model and LanceDB table are loaded once at startup (~15s for large model).
- Text queries are wrapped in `"a photo of {query}"` before encoding — this is intentional and matches CLIP's training distribution.
- Search uses `.bypass_vector_index()` for exact brute-force KNN (correct for this dataset size).
- `/image?path=<abs_path>` serves image files; restricted to `data/` subtree to prevent path traversal.
- Lightbox navigation (prev/next arrows and keyboard ←/→) only steps through cards currently visible above the threshold filter.

## LanceDB / indexing

- **Table:** `images` &nbsp; **Metric:** cosine &nbsp; **Dimensions:** set by `embed_dim` in config
- **Schema:** `path` (str), `filename` (str), `source_dir` (str), `vector` (float32[embed_dim])
- Embeddings are L2-normalised at index time, so cosine similarity equals dot product.
- Use `pyarrow` (not pandas) to inspect: `table.to_arrow().slice(0, n)`
- `db.list_tables()` returns a result object — use `.tables` for membership: `name in db.list_tables().tables`

## Transformers 5.x quirks

- `CLIPModel.get_image_features()` and `get_text_features()` both return `BaseModelOutputWithPooling`, not a tensor. The projected embedding is in `.pooler_output`.
- Suppress noisy model-loading output: `TRANSFORMERS_VERBOSITY=error`
