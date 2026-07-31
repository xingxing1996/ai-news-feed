"""模型配置：读 config/models.yaml（你维护逻辑名），解析成具体版本 id。

- get_model_pool()：返回具体版本 id 列表，优先用 models.yaml 里 resolved 的，
  否则用 ARK_API_KEY 实时查 Ark /models 取最新，再否则兜底。
- resolve_models(api_key)：把每个逻辑名匹配到最新版本 id，写回 models.yaml 的 resolved。
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "https://ark.cn-beijing.volces.com/api/v3"
# 兜底（resolved 为空且无法联网查时）
FALLBACK_POOL = ["glm-5-2-260617", "doubao-seed-2-1-pro-260628"]


def _models_path() -> Path:
    env = os.getenv("MODELS_CONFIG")
    if env and Path(env).exists():
        return Path(env)
    return _ROOT / "config" / "models.yaml"


def load_models_config() -> dict:
    path = _models_path()
    if not path.exists():
        return {"models": [], "resolved": {}, "base_url": DEFAULT_BASE}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_models_config(cfg: dict) -> Path:
    path = _models_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def query_ark_models(api_key: str, base_url: str = DEFAULT_BASE) -> list[str]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {api_key}"}
    )
    with urllib.request.urlopen(req, timeout=25) as r:
        return [m.get("id") for m in json.load(r).get("data", [])]


def resolve_logical(name: str, ids: list[str]) -> str | None:
    """逻辑名 → 最新具体版本 id（前缀/子串匹配，按字符串降序取最新）。"""
    cands = [i for i in ids if i and (i.startswith(name) or name in i)]
    return sorted(cands)[-1] if cands else None


def get_model_pool() -> list[str]:
    """具体版本 id 列表。"""
    cfg = load_models_config()
    names = cfg.get("models") or []
    resolved = cfg.get("resolved") or {}
    pool = [resolved[n] for n in names if resolved.get(n)]
    if pool:
        return pool
    key = os.getenv("ARK_API_KEY")
    if key:
        try:
            ids = query_ark_models(key, cfg.get("base_url", DEFAULT_BASE))
            pool = [r for r in (resolve_logical(n, ids) for n in names) if r]
            if pool:
                return pool
        except Exception:
            pass
    return FALLBACK_POOL


def resolve_models(api_key: str | None = None) -> dict:
    """查询 Ark /models，把每个逻辑名的最新版本写回 models.yaml。返回 {name: id}。"""
    cfg = load_models_config()
    names = cfg.get("models") or []
    key = api_key or os.getenv("ARK_API_KEY")
    if not key:
        return {"ok": False, "err": "无 ARK_API_KEY"}
    ids = query_ark_models(key, cfg.get("base_url", DEFAULT_BASE))
    resolved = {}
    missing = []
    for n in names:
        rid = resolve_logical(n, ids)
        if rid:
            resolved[n] = rid
        else:
            missing.append(n)
    cfg["resolved"] = resolved
    save_models_config(cfg)
    return {"ok": True, "resolved": resolved, "missing": missing, "n_available": len(ids)}
