"""组合指标：年化收益 / 最大回撤 / Sharpe / 换手 / 超额。"""
from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING = 252


def _ann_return(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    total = (1 + returns).prod()
    years = len(returns) / _TRADING
    return float(total ** (1 / years) - 1) if years > 0 else float("nan")


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def _sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    if returns.empty or returns.std() == 0:
        return float("nan")
    excess = returns - rf / _TRADING
    return float(excess.mean() / returns.std() * np.sqrt(_TRADING))


def compute(port_ret: pd.Series, bench_ret: pd.Series, rf: float = 0.0) -> dict:
    port_ret = port_ret.dropna()
    bench_ret = bench_ret.reindex(port_ret.index).fillna(0.0)
    excess = port_ret - bench_ret
    out = {
        "ann_return": _ann_return(port_ret),
        "ann_vol": float(port_ret.std() * np.sqrt(_TRADING)),
        "sharpe": _sharpe(port_ret, rf),
        "max_drawdown": _max_drawdown(port_ret),
        "bench_ann_return": _ann_return(bench_ret),
        "excess_ann": _ann_return(excess),
        "information_ratio": _sharpe(excess, 0.0),
        "win_rate": float((port_ret > 0).mean()) if not port_ret.empty else float("nan"),
        "n_days": int(len(port_ret)),
    }
    return out
