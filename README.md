# MyMedia

Semantic image search over a personal photo library using [CLIP](https://openai.com/research/clip) and [LanceDB](https://lancedb.com/).

Type a natural-language description — *"birthday cake"*, *"dog at the beach"*, *"sunset over mountains"* — and get back the most visually matching photos from your library.

![Search UI](https://raw.githubusercontent.com/rkaunismaa/MyMedia/master/docs/screenshot.png)

---

## How it works

1. **Indexing** — `build_clip_index.py` scans your image directories, runs each image through the CLIP vision encoder, and stores the L2-normalised embeddings in a LanceDB vector database.
2. **Search** — `app.py` encodes your text query through the CLIP text encoder (using the `"a photo of {query}"` prompt template that matches CLIP's training distribution) and performs exact cosine-similarity search against the index.

---

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA (CPU works but indexing will be slow)
- A Python virtual environment with the following packages:

```
torch
transformers
lancedb
fastapi
uvicorn[standard]
Pillow
pyarrow
```

---

## Setup

### 1. Create and populate the virtual environment

```bash
python -m venv .mymedia
uv pip install torch transformers lancedb "fastapi" "uvicorn[standard]" Pillow pyarrow --python .mymedia/bin/python
```

### 2. Add your images

Place image directories under `data/`. By default the project expects:

```
data/iCloudPhotos/
data/iCloudPhotos1/
```

Edit `config.toml` to point at your own directories:

```toml
[index]
data_dirs = ["data/MyPhotos", "data/MorePhotos"]
```

### 3. Configure the model (optional)

`config.toml` controls the CLIP model and embedding dimension:

```toml
[model]
model_id  = "openai/clip-vit-large-patch14"   # or clip-vit-base-patch32 for faster indexing
embed_dim = 768                                # must match the model (512 for base, 768 for large)
```

**Changing the model requires a full index rebuild.**

### 4. Build the index

```bash
TRANSFORMERS_VERBOSITY=error .mymedia/bin/python build_clip_index.py
```

The model is downloaded from Hugging Face on the first run. On an RTX 4090 with `clip-vit-large-patch14`, indexing ~1250 images takes about 2 minutes.

### 5. Start the web app

```bash
TRANSFORMERS_VERBOSITY=error .mymedia/bin/python app.py
```

Open **http://localhost:8000** in your browser.

---

## Using the search UI

- Type a description and press **Enter** or click **Search**
- Use the **Results** dropdown to fetch up to 100 matches
- Drag the **Min score** slider to filter out low-confidence results — the grid updates instantly
- Each image shows its **cosine similarity score** as a badge
- Click any image to open a **lightbox** — use the **‹ ›** arrows or **← →** keyboard keys to navigate, **Esc** to close

### Saving images

- Click the **✓ circle** in the top-left corner of any card to select it (green border)
- Click again to deselect
- Once one or more images are selected, the **Save Images (N)** button activates
- Click it to download all selected images as a single `selected_images.zip`
- Selection resets on each new search

---

## Project structure

```
build_clip_index.py   # Index builder
app.py                # FastAPI web app (UI + search + image serving)
config.toml           # Model and index configuration
data/                 # Image directories (gitignored)
clip_index.lancedb/   # Vector database (gitignored)
.mymedia/             # Python virtual environment (gitignored)
```
