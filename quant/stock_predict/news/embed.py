"""新闻 embedding + FAISS 检索（设计文档第 6 节）。

默认接本地 ``qwen3-embedding-0.6B``（sentence-transformers 格式，你本地已有）。
路径在 settings.embedding.model_path 配置。
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import get_settings

log = logging.getLogger(__name__)

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer  # noqa: WPS433

        cfg = get_settings().embedding
        path = cfg.get("model_path") or None
        _MODEL = SentenceTransformer(path) if path else SentenceTransformer("BAAI/bge-small-zh-v1.5")
    return _MODEL


def embed(texts: list[str]) -> "list[list[float]]":
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def build_index(items: list[dict], index_path: str | None = None):
    """items: [{title, content, ...}] → 建 FAISS 索引并落盘。"""
    import faiss  # noqa: WPS433
    import numpy as np

    cfg = get_settings().embedding
    dim = int(cfg.get("dim", 1024))
    texts = [f"{it.get('title','')} {it.get('content','')}" for it in items]
    if not texts:
        return None
    vecs = np.array(embed(texts), dtype="float32")
    if vecs.shape[1] != dim:
        dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    path = index_path or cfg.get("index_path")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, path)
    return index


def search(query: str, index, k: int = 5):
    import numpy as np

    vec = np.array(embed([query]), dtype="float32")
    scores, idx = index.search(vec, k)
    return idx[0].tolist(), scores[0].tolist()
