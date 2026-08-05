"""因子预处理（机构标准流程）：去极值 + 截面标准化。

⚠️ 性能：全部用向量化/C 加速的 groupby 聚合（transform("mean"/"std")、rank），
不要用 per-date 的 Python lambda（在 14 万行×71 列上会慢到不可用）。

- winsorize：全局按列分位 clip（快，足够防离群）；
- 标准化：截面 z-score（向量化）或 rank(pct)（最稳健，C 加速）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_global(df: pd.DataFrame, q: tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
    """全局按列去极值（向量化，快）。"""
    lo = df.quantile(q[0])
    hi = df.quantile(q[1])
    return df.clip(lower=lo, upper=hi, axis=1)


def winsorize_mad_global(df: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """全局 MAD 去极值（向量化）。"""
    med = df.median()
    mad = (df - med).abs().median()
    bound = k * 1.4826 * mad
    bound = bound.where(bound > 0, np.nan)
    return df.clip(lower=med - bound, upper=med + bound, axis=1)


def zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
    """截面 z-score（用 C 加速的字符串聚合，不用 Python lambda）。"""
    g = df.groupby(level="date")
    mean = g.transform("mean")
    std = g.transform("std")
    return (df - mean).div(std.where(std > 0, 1.0))


def rank_cs(df: pd.DataFrame) -> pd.DataFrame:
    """截面秩归一化（pct rank，C 加速，最稳健、最快）。"""
    return df.groupby(level="date").rank(pct=True)


def process_features(df: pd.DataFrame, method: str = "rank",
                     winsorize: str = "quantile") -> pd.DataFrame:
    """标准因子预处理：去极值 → 标准化（全部向量化）。

    method: "rank"(默认,快) | "zscore" | "none"
    winsorize: "quantile" | "mad" | "none"
    """
    if df.empty:
        return df
    if winsorize == "quantile":
        df = winsorize_global(df)
    elif winsorize == "mad":
        df = winsorize_mad_global(df)
    if method == "rank":
        return rank_cs(df)
    if method == "zscore":
        return zscore_cs(df)
    return df
