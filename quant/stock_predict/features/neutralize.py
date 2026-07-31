"""风格中性化：把因子/收益对「行业 + 市值」做截面回归取残差。

目的：剥离风格暴露，让剩下的 IC 是「纯 alpha」而非吃了某个行业或大盘股的风格。
label 已经是「行业超额」，这里再对 log 市值中性化，得到行业+市值中性 alpha。
也可对特征做中性化（默认对 label 做）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize(y: pd.Series, style: pd.DataFrame, min_n: int = 10) -> pd.Series:
    """逐日把 y 对 style 的列做 OLS，返回残差。

    y: (date, code) Series
    style: (date, code) DataFrame，列如 [log_mcap]（行业可用 one-hot）
    """
    if style is None or style.empty or y.empty:
        return y
    style_cols = [c for c in style.columns if c != "industry"]
    if not style_cols:
        return y
    frame = pd.concat([y.rename("__y__"), style[style_cols]], axis=1)
    frame["__y__"] = pd.to_numeric(frame["__y__"], errors="coerce").astype(float)

    def _resid(g: pd.DataFrame) -> pd.Series:
        sub = g.dropna()
        if len(sub) < len(style_cols) + min_n // 2:
            return g["__y__"].astype(float)
        X = np.column_stack([np.ones(len(sub)), sub[style_cols].values.astype(float)])
        beta, *_ = np.linalg.lstsq(X, sub["__y__"].values.astype(float), rcond=None)
        resid = g["__y__"].astype(float).copy()
        resid.loc[sub.index] = sub["__y__"].values - X @ beta
        return resid

    out = frame.groupby(level="date", group_keys=False).apply(_resid)
    return out.rename(y.name) if y.name else out


def add_size_style(features: pd.DataFrame, market_cap_col: str = "market_cap") -> pd.DataFrame:
    """构造 log 市值风格列（用于中性化）。需要 features 里有 market_cap。"""
    if market_cap_col in features.columns:
        style = pd.DataFrame(index=features.index)
        style["log_mcap"] = np.log(features[market_cap_col].clip(lower=1))
        return style
    return pd.DataFrame(index=features.index)
