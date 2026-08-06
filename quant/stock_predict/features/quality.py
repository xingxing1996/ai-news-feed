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
    fin["report_period"] = pd.to_datetime(fin.get("report_period"), errors="coerce")

    # 同比增长 YoY：用去年同期 report_period 自连接，避免环比(Q4→Q1)的季节性失真。
    if "report_period" in fin.columns and fin["report_period"].notna().any() and len(fin) > 1:
        prev = fin[["code", "report_period", "revenue", "profit"]].copy()
        prev["report_period"] = prev["report_period"] + pd.DateOffset(years=1)  # 去年同期行搬到今年的键
        prev = prev.rename(columns={"revenue": "_rev_yoy", "profit": "_prof_yoy"})
        fin = fin.merge(prev, on=["code", "report_period"], how="left")
        if ("revenue_growth" not in fin.columns) or fin["revenue_growth"].isna().all():
            fin["revenue_growth"] = np.where(
                fin["_rev_yoy"].notna() & (fin["_rev_yoy"] != 0) & fin["revenue"].notna(),
                fin["revenue"] / fin["_rev_yoy"] - 1, np.nan)
        if ("profit_growth" not in fin.columns) or fin["profit_growth"].isna().all():
            fin["profit_growth"] = np.where(
                fin["_prof_yoy"].notna() & (fin["_prof_yoy"] != 0) & fin["profit"].notna(),
                fin["profit"] / fin["_prof_yoy"] - 1, np.nan)
        fin = fin.drop(columns=[c for c in ("_rev_yoy", "_prof_yoy") if c in fin.columns])
    else:
        # 无 report_period（退化）：保留源增长或留空
        if "revenue_growth" not in fin.columns:
            fin["revenue_growth"] = np.nan
        if "profit_growth" not in fin.columns:
            fin["profit_growth"] = np.nan

    # 严格 PIT (Point-in-Time) 时间对齐：交易日 (date) 只能匹配在该交易日之前 (pub_date <= date) 已经发布的最新财报
    d = daily[["date", "code"]].copy()
    d["date_dt"] = pd.to_datetime(d["date"])
    fin["pub_date_dt"] = pd.to_datetime(fin["pub_date"])

    # merge_asof 要求 on 键(date_dt)全局单调；多 code 时按 [code,on] 排序不满足，故仅按 on 键排序
    d_sorted = d.sort_values("date_dt").reset_index(drop=True)
    fin_sorted = fin.sort_values("pub_date_dt").reset_index(drop=True)

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
