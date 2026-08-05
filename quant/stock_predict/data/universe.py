"""股票池解析与入库。

读 ``config/universe.yaml``，按 demo_mode 截断，写入 ``stock`` 表，
并返回统一格式的 DataFrame。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from ..config import PROJECT_ROOT, get_settings
from .models import Stock
from .warehouse import upsert_dataframe

_COUNTRY = {"cn": "中国", "hk": "中国香港", "us": "美国", "kr": "韩国"}


def load_universe_file() -> dict:
    path = PROJECT_ROOT / get_settings().universe.config_file
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_universe(demo: bool | None = None) -> pd.DataFrame:
    """返回统一格式的股票池 DataFrame。

    列：code, name, market, industry, country
    demo 为 None 时取 settings.universe.demo_mode。
    """
    cfg = get_settings()
    demo = cfg.universe.demo_mode if demo is None else demo
    raw = load_universe_file()

    demo_size = dict(cfg.universe.get("demo_size", {})) or {}
    # 按市场过滤（拆分训练用：GHA 只跑 us/kr，中国机器跑 cn/hk）
    only_markets = cfg.universe.get("markets") or ["cn", "hk", "us", "kr"]
    rows: list[dict] = []
    for market in ("cn", "hk", "us", "kr"):
        if market not in only_markets:
            continue
        items = raw.get(market, []) or []
        if demo and market in demo_size:
            items = items[: int(demo_size[market])]
        for it in items:
            rows.append(
                {
                    "code": it["code"],
                    "name": it.get("name"),
                    "market": market,
                    "industry": it.get("industry"),
                    "country": _COUNTRY.get(market),
                }
            )
    return pd.DataFrame(rows)


def universe_to_db(df: pd.DataFrame | None = None) -> int:
    """把股票池写入 stock 表（upsert），返回写入行数。"""
    if df is None:
        df = resolve_universe()
    return upsert_dataframe(df, Stock)


def universe_by_market(df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    df = resolve_universe() if df is None else df
    return {m: g.reset_index(drop=True) for m, g in df.groupby("market")}
