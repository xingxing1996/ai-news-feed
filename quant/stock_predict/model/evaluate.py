"""模型评估（设计文档第 9 节）。

不看准确率，看 **Rank IC**（股票排序能力）。
- Rank IC：每个交易日，预测概率与真实 label 的 Spearman 相关。
- ICIR = mean(IC) / std(IC)。
- 命中率（仅参考）：预测概率 >0.5 中 label=1 的比例。
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rank_ic_by_date(prob: pd.Series, label: pd.Series, date: pd.Series) -> pd.Series:
    df = pd.DataFrame({"prob": prob.values, "label": label.values, "date": date.values}).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    def _ic(g: pd.DataFrame) -> float:
        if g["label"].nunique() < 2:
            return np.nan
        rho, _ = spearmanr(g["prob"], g["label"])
        return float(rho)

    try:
        return df.groupby("date", group_keys=False).apply(_ic, include_groups=False)
    except TypeError:  # 旧版 pandas 无 include_groups
        return df.groupby("date", group_keys=False)[["prob", "label"]].apply(_ic)


def summarize(prob: pd.Series, label: pd.Series, date: pd.Series) -> dict:
    ic = rank_ic_by_date(prob, label, date).dropna()
    hi = prob[label == 1]
    hit = float((hi > 0.5).mean()) if len(hi) else float("nan")
    return {
        "rank_ic_mean": float(ic.mean()) if not ic.empty else float("nan"),
        "rank_ic_std": float(ic.std()) if not ic.empty else float("nan"),
        "icir": float(ic.mean() / ic.std()) if not ic.empty and ic.std() else float("nan"),
        "ic_positive_ratio": float((ic > 0).mean()) if not ic.empty else float("nan"),
        "hit_rate": hit,
        "n_days": int(len(ic)),
    }


def quantile_analysis(pred_df: pd.DataFrame, n_quantiles: int = 5) -> dict:
    """分位组合分析（机构标准因子评估）。

    pred_df 需含 date/code/prob/future_return。
    - 每日按 prob 分 n 组，算各组未来收益均值；
    - 多头(最高分组) - 空头(最低分组) 的年化与 Sharpe；
    - 单调性：分组序号 vs 平均收益的相关（越接近1越单调）。
    """
    df = pred_df.dropna(subset=["prob", "future_return"]).copy()
    if df.empty:
        return {}
    try:
        from ..config import get_settings
        horizon = int(get_settings().feature.get("label_horizon", 20))
    except Exception:  # noqa: BLE001
        horizon = 20

    def _q(x):
        try:
            return pd.qcut(x, n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            return None

    df["q"] = df.groupby("date")["prob"].transform(_q)
    df = df.dropna(subset=["q"])
    if df["q"].nunique() < 2:
        return {}
    qret = df.groupby(["date", "q"])["future_return"].mean().unstack("q").sort_index(axis=1)
    cols = list(qret.columns)
    long_q, short_q = cols[-1], cols[0]
    ls = (qret[long_q] - qret[short_q]).dropna()

    def _ann(s):
        # 未来收益是 horizon 日重叠收益，用 mean*(252/horizon) 粗略年化（避免复利爆炸）
        if s.empty:
            return float("nan")
        return float(s.mean() * 252 / max(horizon, 1))

    grp_ann = {f"q{int(c)}": _ann(qret[c].dropna()) for c in cols}
    mono = float(np.corrcoef(cols, [grp_ann[f"q{int(c)}"] for c in cols])[0, 1]) if len(cols) > 2 else float("nan")
    return {
        "n_quantiles": int(len(cols)),
        "long_short_ann": _ann(ls),
        "long_short_sharpe": float(ls.mean() / ls.std() * np.sqrt(252)) if ls.std() else float("nan"),
        "long_ann": _ann(qret[long_q].dropna()),
        "short_ann": _ann(qret[short_q].dropna()),
        "monotonicity": mono,
        "group_ann_returns": grp_ann,
    }


def ic_decay(pred_df: pd.DataFrame, daily: pd.DataFrame, lags: list[int] | None = None) -> dict:
    """IC 衰减：预测概率与未来 1/5/10/20 日收益的 Rank IC，看信号有效期。

    daily 需含 date/code/close（用于算各 lag 的未来收益）。
    """
    lags = lags or [1, 5, 10, 20]
    piv = daily.pivot(index="date", columns="code", values="close").sort_index()
    out = {}
    p = pred_df.dropna(subset=["prob"]).copy()
    for lag in lags:
        fut = piv.pct_change(lag).shift(-lag)  # t 时刻看 t..t+lag 的收益
        fut = fut.stack().rename("fut").reset_index().rename(columns={"level_0": "date", "level_1": "code"})
        fut["date"] = fut["date"].astype(str)
        m = p.merge(fut, on=["date", "code"], how="inner").dropna(subset=["prob", "fut"])
        if m.empty:
            out[f"ic_{lag}d"] = float("nan")
            continue
        from scipy.stats import spearmanr
        ic = m.groupby("date").apply(lambda g: spearmanr(g["prob"], g["fut"])[0] if g["fut"].nunique() > 1 else np.nan).dropna()
        out[f"ic_{lag}d"] = float(ic.mean()) if not ic.empty else float("nan")
    return out
