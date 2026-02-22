# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Python venv:** `.mymedia/` — always invoke as `.mymedia/bin/python`
- **GPU:** NVIDIA GeForce RTX 4090 with CUDA 12.6
- **Key packages:** torch 2.10.0+cu126, lancedb 0.29.2, transformers 5.2.0, pyarrow (no pandas installed)

## Running scripts

```bash
# Standard invocation (suppresses noisy model-loading progress bars)
TRANSFORMERS_VERBOSITY=error .mymedia/bin/python <script.py>

# Rebuild the CLIP vector index from scratch
TRANSFORMERS_VERBOSITY=error .mymedia/bin/python build_clip_index.py
```

## Project structure

| Path | Purpose |
|---|---|
| `build_clip_index.py` | Scans image dirs, runs CLIP inference, writes LanceDB index |
| `config.toml` | All tuneable parameters (dirs, model, batch size, etc.) |
| `data/iCloudPhotos/` | First iCloud photo export (~800 files) |
| `data/iCloudPhotos1/` | Second iCloud photo export (~880 files) |
| `clip_index.lancedb/` | Persisted LanceDB vector database |
| `.mymedia/` | Python virtual environment |

## LanceDB index

- **Table:** `images`  **Dimensions:** 512  **Metric:** cosine
- **Schema:** `path` (str), `filename` (str), `source_dir` (str), `vector` (float32[512])
- Embeddings are L2-normalised, so cosine similarity equals dot product.
- Use `pyarrow` (not pandas) to inspect table contents: `table.to_arrow().slice(0, n)`

## Transformers 5.x quirks

- `CLIPModel.get_image_features()` returns `BaseModelOutputWithPooling`, not a tensor. Extract the embedding via `.pooler_output` (already 512-dim, already projected).
- `db.table_names()` is deprecated — use `db.list_tables()`.
- `db.list_tables()` returns a result object, not a list — membership checks require `.tables`: `name in db.list_tables().tables`.
