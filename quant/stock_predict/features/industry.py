"""行业因子（设计文档 5.4）：industry_score。

设计文档把「行业周期」（DRAM价格/HBM需求/汽车销量/库存周期）列为你需要自己接入的优势数据。
Phase 1 提供两层：
  1) 默认：用行业相对动量（industry_momentum）作占位因子——真实、可计算。
  2) 外部接入：传 external_csv（列 industry / date / industry_score）即可合并行业周期得分。

industry_momentum = 行业近 20 日平均收益 − 全市场近 20 日平均收益。
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_MOM_WIN = 20


def compute_industry_factors(
    daily: pd.DataFrame,
    universe: pd.DataFrame,
    external_csv: str | None = None,
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()

    ind_map = universe.set_index("code")["industry"].to_dict()
    d = daily[["date", "code", "close"]].copy()
    d["industry"] = d["code"].map(ind_map)
    d = d.dropna(subset=["industry"])
    d = d.sort_values(["code", "date"])
    d["ret"] = d.groupby("code")["close"].pct_change()

    # 行业日收益
    ind_ret = d.groupby(["date", "industry"])["ret"].mean().reset_index().rename(columns={"ret": "ind_ret"})
    mkt_ret = d.groupby("date")["ret"].mean().reset_index().rename(columns={"ret": "mkt_ret"})
    ind_ret = ind_ret.merge(mkt_ret, on="date", how="left")
    ind_ret = ind_ret.sort_values(["industry", "date"])
    ind_ret["ind_mom"] = ind_ret.groupby("industry")["ind_ret"].transform(
        lambda s: s.rolling(_MOM_WIN, min_periods=_MOM_WIN // 2).mean()
    )
    ind_ret["mkt_mom"] = ind_ret.groupby("industry")["mkt_ret"].transform(
        lambda s: s.rolling(_MOM_WIN, min_periods=_MOM_WIN // 2).mean()
    )
    ind_ret["industry_momentum"] = ind_ret["ind_mom"] - ind_ret["mkt_mom"]

    out = d[["date", "code", "industry"]].merge(
        ind_ret[["date", "industry", "industry_momentum"]], on=["date", "industry"], how="left"
    )
    out = out.set_index(["date", "code"]).sort_index()
    result = out[["industry_momentum"]]
    result["industry"] = out["industry"]  # 保留分类列，供模型/解释用

    # 外部行业周期得分（可选）
    if external_csv and Path(external_csv).exists():
        ext = pd.read_csv(external_csv)
        if {"industry", "date", "industry_score"}.issubset(ext.columns):
            ext["date"] = pd.to_datetime(ext["date"]).dt.strftime("%Y-%m-%d")
            ext = ext.set_index(["date", "industry"])["industry_score"]
            result = result.join(ext, on=["date", "industry"], how="left")
            log.info("[industry] 合并外部 industry_score: %s", external_csv)

    log.info("[industry] 行业因子数=%d（不含分类列）", sum(1 for c in result.columns if c != "industry"))
    return result
