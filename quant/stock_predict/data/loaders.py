"""数据采集编排：按市场路由到对应加载器，汇总写入 SQLite + Parquet。

入口：``fetch_and_store(universe_df, settings)``。
真实数据：A股→akshare_loader，港/美→yfinance_loader。
兜底：synthetic=True 时全部走合成数据（无网络/测试/演示）。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from ..config import AttrDict, get_settings
from .models import DailyPrice, Financial
from .warehouse import upsert_dataframe, write_parquet

log = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kw):  # type: ignore
        return it


def _date_range(cfg: AttrDict) -> tuple[str, str]:
    years = int(cfg.data.get("years_back", 6))
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=years * 365 + 30)).strftime("%Y-%m-%d")
    return start, end


def _get_loader(market: str, synthetic: bool):
    if synthetic:
        from . import synthetic_loader as L
        return L
    if market in ("cn", "hk"):
        from . import akshare_loader as L  # A股 + 港股都走 AKShare（国内稳定）
    else:
        from . import yfinance_loader as L  # 美股/韩股：yfinance（国内可能需代理）
    return L


def fetch_all(universe_df: pd.DataFrame, start: str, end: str, synthetic: bool) -> dict[str, pd.DataFrame]:
    """对整个股票池拉数据，返回 {daily, valuation, financial} 三个大表。"""
    daily_parts, val_parts, fin_parts = [], [], []
    rows = list(universe_df.itertuples(index=False))
    for r in tqdm(rows, desc="ingest", disable=len(rows) <= 3):
        L = _get_loader(r.market, synthetic)
        is_ak = (not synthetic) and r.market in ("cn", "hk")
        d = L.fetch_daily(r.code, start, end, r.market) if is_ak else L.fetch_daily(r.code, start, end)
        if not d.empty:
            daily_parts.append(d)
        if hasattr(L, "fetch_valuation"):
            v = L.fetch_valuation(r.code, start, end, r.market) if is_ak else L.fetch_valuation(r.code, start, end)
            if not v.empty:
                val_parts.append(v)
        f = L.fetch_financial(r.code)
        if not f.empty:
            fin_parts.append(f)

    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    valuation = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    financial = pd.concat(fin_parts, ignore_index=True) if fin_parts else pd.DataFrame()
    return {"daily": daily, "valuation": valuation, "financial": financial}


def fetch_and_store(universe_df: pd.DataFrame | None = None, settings: AttrDict | None = None) -> dict[str, Any]:
    """采集并入库。返回统计信息。"""
    cfg = settings or get_settings()
    from .universe import resolve_universe

    if universe_df is None:
        universe_df = resolve_universe()

    start, end = _date_range(cfg)
    synthetic = bool(cfg.data.get("synthetic", False))
    log.info("采集区间 %s ~ %s, synthetic=%s, 标的数=%d", start, end, synthetic, len(universe_df))

    data = fetch_all(universe_df, start, end, synthetic)

    stats: dict[str, Any] = {"start": start, "end": end, "synthetic": synthetic}

    # daily_price → SQLite + Parquet
    if not data["daily"].empty:
        upsert_dataframe(data["daily"], DailyPrice)
        write_parquet(data["daily"], "daily_price")
    stats["daily_rows"] = len(data["daily"])

    # valuation → Parquet（点在时间上的 pe/pb）
    if not data["valuation"].empty:
        write_parquet(data["valuation"], "valuation")
    stats["valuation_rows"] = len(data["valuation"])

    # financial → SQLite
    if not data["financial"].empty:
        upsert_dataframe(data["financial"], Financial)
    stats["financial_rows"] = len(data["financial"])

    log.info("入库完成: %s", stats)
    return stats
