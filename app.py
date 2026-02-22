#!/usr/bin/env python3
"""
Web app: text-to-image search over the CLIP LanceDB index.
Run:  TRANSFORMERS_VERBOSITY=error .mymedia/bin/python app.py
Then open: http://localhost:8000
"""

import tomllib
from pathlib import Path

import lancedb
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import CLIPModel, CLIPProcessor

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "config.toml", "rb") as _f:
    _cfg = tomllib.load(_f)

DB_PATH    = BASE_DIR / _cfg["index"]["db_path"]
TABLE_NAME = _cfg["index"]["table_name"]
MODEL_ID   = _cfg["model"]["model_id"]

# ── Load model & DB at startup ────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP model on {device} …")
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model     = CLIPModel.from_pretrained(MODEL_ID).to(device)
model.eval()
print("Model ready.")

db    = lancedb.connect(str(DB_PATH))
table = db.open_table(TABLE_NAME)
print(f"LanceDB table '{TABLE_NAME}' open — {table.count_rows()} rows.")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="MyMedia Image Search")


@torch.inference_mode()
def embed_text(query: str) -> list[float]:
    inputs  = processor(text=[query], return_tensors="pt", padding=True).to(device)
    outputs = model.get_text_features(**inputs)
    features = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().float().squeeze().tolist()


@app.get("/search")
def search(q: str = Query(..., min_length=1), n: int = Query(20, ge=1, le=100)):
    """Return the top-n images most similar to the text query."""
    vector = embed_text(q)
    results = (
        table.search(vector)
             .limit(n)
             .select(["path", "filename", "source_dir"])
             .to_arrow()
    )
    hits = [
        {
            "filename":   results.column("filename")[i].as_py(),
            "source_dir": results.column("source_dir")[i].as_py(),
            "path":       results.column("path")[i].as_py(),
        }
        for i in range(len(results))
    ]
    return {"query": q, "results": hits}


@app.get("/image")
def serve_image(path: str):
    """Serve an image by its absolute path (must be inside DATA_DIRS)."""
    p = Path(path).resolve()
    # Safety: only serve files inside the data directory
    data_root = (BASE_DIR / "data").resolve()
    if not str(p).startswith(str(data_root)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(p))


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


# ── Frontend ──────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MyMedia Search</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: system-ui, sans-serif;
    background: #0f0f0f;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  header {
    width: 100%;
    padding: 2rem 1rem 1.5rem;
    text-align: center;
    border-bottom: 1px solid #222;
  }

  header h1 { font-size: 1.6rem; font-weight: 600; letter-spacing: .05em; color: #fff; }
  header p  { margin-top: .35rem; font-size: .85rem; color: #666; }

  .search-bar {
    display: flex;
    gap: .5rem;
    margin-top: 1.2rem;
    justify-content: center;
  }

  .search-bar input {
    width: min(520px, 80vw);
    padding: .6rem 1rem;
    border-radius: 8px;
    border: 1px solid #333;
    background: #1a1a1a;
    color: #fff;
    font-size: 1rem;
    outline: none;
    transition: border-color .2s;
  }
  .search-bar input:focus { border-color: #555; }

  .search-bar button {
    padding: .6rem 1.2rem;
    border-radius: 8px;
    border: none;
    background: #3a7bd5;
    color: #fff;
    font-size: 1rem;
    cursor: pointer;
    transition: background .2s;
  }
  .search-bar button:hover    { background: #2f66b8; }
  .search-bar button:disabled { background: #333; cursor: default; }

  .meta {
    margin-top: 1rem;
    font-size: .8rem;
    color: #555;
    min-height: 1.2em;
  }

  #grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 8px;
    padding: 1.5rem;
    width: 100%;
    max-width: 1600px;
  }

  .card {
    position: relative;
    overflow: hidden;
    border-radius: 8px;
    background: #1a1a1a;
    cursor: pointer;
    aspect-ratio: 1;
  }

  .card img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transition: transform .25s;
  }
  .card:hover img { transform: scale(1.04); }

  .card .label {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    padding: .35rem .5rem;
    background: linear-gradient(transparent, rgba(0,0,0,.75));
    font-size: .7rem;
    color: #ccc;
    opacity: 0;
    transition: opacity .2s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .card:hover .label { opacity: 1; }

  /* Lightbox */
  #lightbox {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,.88);
    z-index: 100;
    align-items: center;
    justify-content: center;
    cursor: zoom-out;
  }
  #lightbox.open { display: flex; }
  #lightbox img {
    max-width: 92vw;
    max-height: 92vh;
    border-radius: 6px;
    box-shadow: 0 8px 40px rgba(0,0,0,.6);
    cursor: default;
  }
  #lightbox-close {
    position: absolute;
    top: 1rem; right: 1.2rem;
    font-size: 2rem;
    color: #fff;
    cursor: pointer;
    line-height: 1;
    user-select: none;
  }

  .spinner {
    display: none;
    width: 32px; height: 32px;
    border: 3px solid #333;
    border-top-color: #3a7bd5;
    border-radius: 50%;
    animation: spin .7s linear infinite;
    margin: 3rem auto;
  }
  .spinner.active { display: block; }
  @keyframes spin { to { transform: rotate(360deg); } }

  #results-count { text-align: center; color: #555; font-size: .85rem; margin: 1rem 0; }

  #n-select {
    background: #1a1a1a;
    border: 1px solid #333;
    color: #ccc;
    border-radius: 6px;
    padding: .3rem .5rem;
    font-size: .85rem;
    cursor: pointer;
  }
</style>
</head>
<body>

<header>
  <h1>MyMedia Image Search</h1>
  <p>CLIP-powered semantic search over your photo library</p>
  <div class="search-bar">
    <input id="query" type="text" placeholder="e.g. sunset over mountains, birthday cake, dog playing…" autofocus>
    <button id="search-btn" onclick="doSearch()">Search</button>
  </div>
  <div class="meta">
    Results:
    <select id="n-select">
      <option value="20">20</option>
      <option value="40">40</option>
      <option value="60">60</option>
      <option value="100">100</option>
    </select>
    &nbsp;<span id="status"></span>
  </div>
</header>

<div class="spinner" id="spinner"></div>
<div id="grid"></div>

<div id="lightbox" onclick="closeLightbox()">
  <span id="lightbox-close" onclick="closeLightbox()">&#x2715;</span>
  <img id="lightbox-img" src="" onclick="event.stopPropagation()">
</div>

<script>
  const input   = document.getElementById('query');
  const grid    = document.getElementById('grid');
  const spinner = document.getElementById('spinner');
  const status  = document.getElementById('status');
  const btn     = document.getElementById('search-btn');
  const lb      = document.getElementById('lightbox');
  const lbImg   = document.getElementById('lightbox-img');

  input.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

  async function doSearch() {
    const q = input.value.trim();
    if (!q) return;
    const n = document.getElementById('n-select').value;

    grid.innerHTML = '';
    spinner.classList.add('active');
    btn.disabled = true;
    status.textContent = '';

    try {
      const res  = await fetch(`/search?q=${encodeURIComponent(q)}&n=${n}`);
      const data = await res.json();

      spinner.classList.remove('active');
      btn.disabled = false;

      if (!data.results || data.results.length === 0) {
        status.textContent = 'No results.';
        return;
      }

      status.textContent = `${data.results.length} result${data.results.length !== 1 ? 's' : ''}`;

      data.results.forEach(hit => {
        const imgUrl = `/image?path=${encodeURIComponent(hit.path)}`;
        const card   = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
          <img src="${imgUrl}" loading="lazy" alt="${hit.filename}">
          <div class="label">${hit.filename}</div>
        `;
        card.querySelector('img').addEventListener('click', () => openLightbox(imgUrl));
        grid.appendChild(card);
      });

    } catch (err) {
      spinner.classList.remove('active');
      btn.disabled = false;
      status.textContent = 'Error: ' + err.message;
    }
  }

  function openLightbox(src) {
    lbImg.src = src;
    lb.classList.add('open');
  }
  function closeLightbox() {
    lb.classList.remove('open');
    lbImg.src = '';
  }
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
