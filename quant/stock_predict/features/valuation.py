"""估值因子（设计文档 5.2）。

核心：用**历史分位**而非绝对值。
  pe_percentile / pb_percentile / fcf_yield / fcf_yield_percentile
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_PCT_WINDOW = 252 * 3  # 3 年滚动分位窗口


def _rolling_pct_rank(s: pd.Series, w: int) -> pd.Series:
    res = s.rolling(w, min_periods=1).rank(pct=True)
    if res.isna().any():
        res = res.ffill().bfill()
    return res


def compute_valuation_factors(
    daily: pd.DataFrame, valuation: pd.DataFrame, financial: pd.DataFrame
) -> pd.DataFrame:
    """返回 (date, code) 索引的估值因子。无估值数据时相应列为 NaN。"""
    if daily.empty:
        return pd.DataFrame()

    feats: list[pd.DataFrame] = []

    # ---- pe / pb 分位（来自 valuation 日序列）+ pe/pb 原值（供日报展示）----
    if not valuation.empty:
        v = valuation.sort_values(["code", "date"]).copy()
        parts = []
        for code, g in v.groupby("code"):
            d = g.copy()
            if "pe" in d and d["pe"].notna().any():
                d["pe_percentile"] = _rolling_pct_rank(d["pe"], _PCT_WINDOW)
            if "pb" in d and d["pb"].notna().any():
                d["pb_percentile"] = _rolling_pct_rank(d["pb"], _PCT_WINDOW)
            parts.append(d)
        v = pd.concat(parts, ignore_index=True)
        v = v.set_index(["date", "code"]).sort_index()
        # 保留原值 + 分位
        cols = [c for c in ["pe", "pb", "pe_percentile", "pb_percentile"] if c in v.columns]
        feats.append(v[cols])


    # ---- fcf_yield（来自 financial 现金流 + daily 市值）----
    if not financial.empty and "market_cap" in daily.columns:
        fin = financial[["code", "pub_date", "cashflow"]].copy()
        fin["pub_date"] = pd.to_datetime(fin["pub_date"])
        fin = fin.sort_values(["code", "pub_date"]).drop_duplicates("code", keep="last")
        d = daily[["date", "code", "market_cap"]].copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.merge(fin[["code", "cashflow"]], on="code", how="left")
        d["fcf_yield"] = d["cashflow"] / d["market_cap"].replace(0, np.nan)
        d = d.sort_values(["code", "date"])
        d["fcf_yield_percentile"] = d.groupby("code")["fcf_yield"].transform(
            lambda s: _rolling_pct_rank(s, _PCT_WINDOW)
        )
        d["date"] = d["date"].dt.strftime("%Y-%m-%d")
        d = d.set_index(["date", "code"]).sort_index()
        feats.append(d[["fcf_yield", "fcf_yield_percentile"]])

    if not feats:
        return pd.DataFrame()
    out = pd.concat(feats, axis=1).sort_index()
    log.info("[valuation] 估值因子数=%d", out.shape[1])
    return out
