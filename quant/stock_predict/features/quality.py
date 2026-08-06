"""财务质量因子（设计文档 5.3）：ROE / 利润增长 / 收入增长 / 毛利率 / 自由现金流。

财务以快照形式挂在每个交易日上（按发布日 ffill，避免未来函数）。
若有多期财报，额外计算增长。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def compute_quality_factors(daily: pd.DataFrame, financial: pd.DataFrame) -> pd.DataFrame:
    if daily.empty or financial.empty:
        return pd.DataFrame()

    fin = financial.copy()
    fin["pub_date"] = pd.to_datetime(fin["pub_date"], errors="coerce")

    # 增长（需要多期）：按 code 报告期排序后 pct_change
    fin = fin.sort_values(["code", "pub_date"])
    growth = []
    for code, g in fin.groupby("code"):
        gg = g.copy()
        if len(gg) > 1:
            if "revenue_growth" not in gg.columns or gg["revenue_growth"].isna().all():
                gg["revenue_growth"] = gg["revenue"].pct_change(fill_method=None)
            if "profit_growth" not in gg.columns or gg["profit_growth"].isna().all():
                gg["profit_growth"] = gg["profit"].pct_change(fill_method=None)
        else:
            if "revenue_growth" not in gg.columns:
                gg["revenue_growth"] = np.nan
            if "profit_growth" not in gg.columns:
                gg["profit_growth"] = np.nan
        growth.append(gg)
    fin = pd.concat(growth, ignore_index=True) if growth else fin

    # 严格 PIT (Point-in-Time) 时间对齐：交易日 (date) 只能匹配在该交易日之前 (pub_date <= date) 已经发布的最新财报
    d = daily[["date", "code"]].copy()
    d["date_dt"] = pd.to_datetime(d["date"])
    fin["pub_date_dt"] = pd.to_datetime(fin["pub_date"])

    d_sorted = d.sort_values(["code", "date_dt"]).reset_index(drop=True)
    fin_sorted = fin.sort_values(["code", "pub_date_dt"]).reset_index(drop=True)

    merged = pd.merge_asof(
        d_sorted,
        fin_sorted,
        by="code",
        left_on="date_dt",
        right_on="pub_date_dt",
        direction="backward"  # 严格只取发布日 <= 当日交易日的已公开财报
    )

    d = merged.copy()
    # 规模 / 利润率
    if "revenue" in d.columns:
        d["log_revenue"] = np.log(d["revenue"].clip(lower=1))
    if "profit" in d.columns and "revenue" in d.columns:
        d["profit_margin"] = d["profit"] / d["revenue"].replace(0, np.nan)

    d = d.set_index(["date", "code"]).sort_index()

    # 截面质量综合分（当日各股票 ROE/毛利/增长的均值排名）
    rank_cols = [c for c in ["roe", "gross_margin", "revenue_growth", "profit_growth"] if c in d.columns]
    if rank_cols:
        ranks = d.groupby(level="date")[rank_cols].rank(pct=True)
        d["quality_rank"] = ranks.mean(axis=1)

    keep = [c for c in ["roe", "gross_margin", "revenue_growth", "profit_growth",
                        "log_revenue", "profit_margin", "quality_rank"] if c in d.columns]
    log.info("[quality] 质量因子数=%d", len(keep))
    return d[keep]
