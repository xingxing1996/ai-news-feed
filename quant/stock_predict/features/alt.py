"""另类数据因子（A股）：北向资金（沪深港通持股变化 = 外资/聪明钱流向）。

北向资金是 A股 个人能拿到的、较有效的差异化 alpha：
  - north_chg5 / north_chg20：持股数量 5/20 日变化率（外资短期流入/流出）
  - north_level：持股数量的历史分位（外资持仓高低）
仅 A股 有该数据；港股/美股为空（对应因子 NaN）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_northbound_factors(northbound: pd.DataFrame) -> pd.DataFrame:
    """北向持股数量 → 变化率 + 历史分位。返回 (date, code) 索引。"""
    if northbound is None or northbound.empty:
        return pd.DataFrame()
    nb = northbound.copy()
    nb["date"] = pd.to_datetime(nb["date"])
    nb["north_shares"] = pd.to_numeric(nb["north_shares"], errors="coerce")
    nb = nb.dropna(subset=["north_shares"])
    nb = nb[nb["north_shares"] > 0].sort_values(["code", "date"])

    g = nb.groupby("code")["north_shares"]
    nb["north_chg5"] = g.pct_change(5)
    nb["north_chg20"] = g.pct_change(20)
    nb = nb.replace([np.inf, -np.inf], np.nan)
    nb["north_level"] = g.rank(pct=True)  # 持股相对历史水平

    nb["date"] = nb["date"].dt.strftime("%Y-%m-%d")
    return nb.set_index(["date", "code"]).sort_index()[["north_chg5", "north_chg20", "north_level"]]


def compute_capital_chip_factors(fund_flow: pd.DataFrame, cyq: pd.DataFrame) -> pd.DataFrame:
    """主力资金净流入占比 + 筹码获利盘/集中度因子。返回 (date, code) 索引。"""
    feats = []
    if fund_flow is not None and not fund_flow.empty:
        fl = fund_flow.copy()
        fl["date"] = pd.to_datetime(fl["date"])
        fl = fl.sort_values(["code", "date"])
        g = fl.groupby("code")
        fl["main_fund_5d"] = g["main_fund_ratio"].rolling(5, min_periods=2).mean()
        fl["super_fund_5d"] = g["super_fund_ratio"].rolling(5, min_periods=2).mean()
        fl["date"] = fl["date"].dt.strftime("%Y-%m-%d")
        fl = fl.set_index(["date", "code"]).sort_index()
        cols = [c for c in ["main_fund_ratio", "super_fund_ratio", "main_fund_5d", "super_fund_5d"] if c in fl.columns]
        feats.append(fl[cols])

    if cyq is not None and not cyq.empty:
        cq = cyq.copy()
        cq["date"] = pd.to_datetime(cq["date"])
        cq = cq.sort_values(["code", "date"])
        cq["date"] = cq["date"].dt.strftime("%Y-%m-%d")
        cq = cq.set_index(["date", "code"]).sort_index()
        cols = [c for c in ["chip_profit_ratio", "chip_concentration_90"] if c in cq.columns]
        feats.append(cq[cols])

    if not feats:
        return pd.DataFrame()
    out = pd.concat(feats, axis=1).sort_index()
    out = out.loc[:, ~out.columns.duplicated()]
    return out
