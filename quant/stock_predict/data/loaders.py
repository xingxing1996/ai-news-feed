"""数据采集编排：按市场路由到对应加载器，汇总写入 SQLite + Parquet。

入口：``fetch_and_store(universe_df, settings)``。
真实数据：A股→akshare_loader，港/美→yfinance_loader。
兜底：synthetic=True 时全部走合成数据（无网络/测试/演示）。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from ..config import AttrDict, get_settings
from .models import DailyPrice, Financial
from .warehouse import read_parquet, upsert_dataframe, write_parquet

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
    try:
        from ..config import get_settings
        sources = get_settings().data.get("sources", {})
        source_name = sources.get(market)
        if source_name == "yfinance":
            from . import yfinance_loader as L
            return L
        if source_name == "akshare":
            from . import akshare_loader as L
            return L
    except Exception:  # noqa: BLE001
        pass

    # 默认路由：cn 走 akshare；hk / us / kr 均走 yfinance (极速直连)
    if market == "cn":
        from . import akshare_loader as L
        return L
    from . import yfinance_loader as L
    return L


def fetch_all(universe_df: pd.DataFrame, start: str, end: str, synthetic: bool,
              start_by_code: dict | None = None) -> dict[str, pd.DataFrame]:
    """对整个股票池拉数据，返回 {daily, valuation, financial} 三个大表。

    start_by_code: {code: 起始日}，增量采集时每只股票只拉「已有最后日期之后」。
    """
    daily_parts, val_parts, fin_parts, nb_parts, flow_parts, cyq_parts = [], [], [], [], [], []
    rows = list(universe_df.itertuples(index=False))
    try:
        from ..config import get_settings as _gs
        fetch_delay = float(_gs().data.get("fetch_delay", 0.5))
    except Exception:  # noqa: BLE001
        fetch_delay = 0.5
    for r in tqdm(rows, desc="ingest", disable=len(rows) <= 3):
        L = _get_loader(r.market, synthetic)
        is_ak = (not synthetic) and r.market in ("cn", "hk")
        s = (start_by_code or {}).get(r.code, start)  # 增量起始日
        d = L.fetch_daily(r.code, s, end, r.market) if is_ak else L.fetch_daily(r.code, s, end)
        if not d.empty:
            daily_parts.append(d)
        if hasattr(L, "fetch_valuation"):
            v = L.fetch_valuation(r.code, s, end, r.market) if is_ak else L.fetch_valuation(r.code, s, end)
            if not v.empty:
                val_parts.append(v)
        if hasattr(L, "fetch_northbound") and (not synthetic) and r.market == "cn":
            nb = L.fetch_northbound(r.code, s, end, r.market)
            if not nb.empty:
                nb_parts.append(nb)
        if hasattr(L, "fetch_fund_flow") and (not synthetic) and r.market == "cn":
            fl = L.fetch_fund_flow(r.code, s, end, r.market)
            if not fl.empty:
                flow_parts.append(fl)
        if hasattr(L, "fetch_cyq") and (not synthetic) and r.market == "cn":
            cq = L.fetch_cyq(r.code, s, end, r.market)
            if not cq.empty:
                cyq_parts.append(cq)
        f = L.fetch_financial(r.code, r.market) if is_ak else L.fetch_financial(r.code)
        if not f.empty:
            fin_parts.append(f)
        if fetch_delay and not synthetic:
            time.sleep(fetch_delay)

    daily_parts = [df for df in daily_parts if not df.empty]
    val_parts = [df for df in val_parts if not df.empty]
    fin_parts = [df for df in fin_parts if not df.empty]
    nb_parts = [df for df in nb_parts if not df.empty]
    flow_parts = [df for df in flow_parts if not df.empty]
    cyq_parts = [df for df in cyq_parts if not df.empty]

    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    valuation = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    financial = pd.concat(fin_parts, ignore_index=True) if fin_parts else pd.DataFrame()
    northbound = pd.concat(nb_parts, ignore_index=True) if nb_parts else pd.DataFrame()
    fund_flow = pd.concat(flow_parts, ignore_index=True) if flow_parts else pd.DataFrame()
    cyq = pd.concat(cyq_parts, ignore_index=True) if cyq_parts else pd.DataFrame()
    return {"daily": daily, "valuation": valuation, "financial": financial,
            "northbound": northbound, "fund_flow": fund_flow, "cyq": cyq}


def fetch_and_store(universe_df: pd.DataFrame | None = None, settings: AttrDict | None = None) -> dict[str, Any]:
    """采集并入库。返回统计信息。"""
    cfg = settings or get_settings()
    from .universe import resolve_universe

    if universe_df is None:
        universe_df = resolve_universe()

    start, end = _date_range(cfg)
    synthetic = bool(cfg.data.get("synthetic", False))
    incremental = bool(cfg.data.get("incremental", False)) and not synthetic

    existing_daily = read_parquet("daily_price") if incremental else pd.DataFrame()
    existing_val = read_parquet("valuation") if incremental else pd.DataFrame()
    start_by_code = None
    if incremental and not existing_daily.empty:
        existing_daily["date"] = existing_daily["date"].astype(str)
        last = existing_daily.groupby("code")["date"].max().to_dict()
        start_by_code = {
            c: (pd.to_datetime(d) + timedelta(days=1)).strftime("%Y-%m-%d") for c, d in last.items()
        }
        log.info("增量采集：%d 只已有数据，仅拉各自最新日之后", len(start_by_code))

    log.info("采集区间 %s ~ %s, synthetic=%s, incremental=%s, 标的数=%d",
             start, end, synthetic, incremental, len(universe_df))
    data = fetch_all(universe_df, start, end, synthetic, start_by_code=start_by_code)

    # 增量合并去重
    daily = data["daily"]
    valuation = data["valuation"]
    if incremental:
        if not existing_daily.empty:
            daily = pd.concat([existing_daily, daily], ignore_index=True) if not daily.empty else existing_daily
            daily = daily.drop_duplicates(["date", "code"], keep="last")
        if not existing_val.empty:
            valuation = pd.concat([existing_val, valuation], ignore_index=True) if not valuation.empty else existing_val
            valuation = valuation.drop_duplicates(["date", "code"], keep="last")

    stats: dict[str, Any] = {"start": start, "end": end, "synthetic": synthetic, "incremental": incremental}

    # daily_price → SQLite + Parquet
    if not daily.empty:
        upsert_dataframe(daily, DailyPrice)
        write_parquet(daily, "daily_price")
    stats["daily_rows"] = len(daily)

    # valuation → Parquet（点在时间上的 pe/pb）
    if not valuation.empty:
        write_parquet(valuation, "valuation")
    stats["valuation_rows"] = len(valuation)

    # northbound → Parquet（北向资金持股，A股另类 alpha）
    if not data.get("northbound", pd.DataFrame()).empty:
        write_parquet(data["northbound"], "northbound")
    stats["northbound_rows"] = len(data.get("northbound", pd.DataFrame()))

    # fund_flow → Parquet（A股主力/超大单资金流向）
    if not data.get("fund_flow", pd.DataFrame()).empty:
        write_parquet(data["fund_flow"], "fund_flow")
    stats["fund_flow_rows"] = len(data.get("fund_flow", pd.DataFrame()))

    # cyq → Parquet（A股筹码分布：获利盘与集中度）
    if not data.get("cyq", pd.DataFrame()).empty:
        write_parquet(data["cyq"], "cyq")
    stats["cyq_rows"] = len(data.get("cyq", pd.DataFrame()))

    # financial → SQLite
    if not data["financial"].empty:
        upsert_dataframe(data["financial"], Financial)
    stats["financial_rows"] = len(data["financial"])

    log.info("入库完成: %s", stats)
    return stats
