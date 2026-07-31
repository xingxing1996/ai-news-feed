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


def explain(m: dict) -> list[str]:
    """把回测指标转成一句句可读结论（供日报/终端展示）。"""
    lines = []
    if not m:
        return lines
    if m.get("ann_return") is not None:
        lines.append(f"策略年化收益 {m['ann_return']:.1%}")
    if m.get("bench_ann_return") is not None:
        lines.append(f"同期基准（市场等权）年化 {m['bench_ann_return']:.1%}")
    if m.get("excess_ann") is not None and m.get("bench_ann_return") is not None:
        sign = "跑赢" if m["excess_ann"] >= 0 else "跑输"
        lines.append(f"相对基准年化超额 {abs(m['excess_ann']):.1%}（{sign}）")
    if m.get("sharpe") is not None:
        s = m["sharpe"]
        grade = "优秀（>2）" if s >= 2 else "良好（1~2）" if s >= 1 else "一般（0~1）" if s >= 0 else "为负"
        lines.append(f"夏普比率 {s:.2f}（{grade}：每承担1份波动换来的超额收益）")
    if m.get("max_drawdown") is not None:
        lines.append(f"历史最大回撤 {m['max_drawdown']:.1%}（从峰值到谷底最深跌幅）")
    if m.get("win_rate") is not None:
        lines.append(f"持仓日胜率 {m['win_rate']:.1%}")
    if m.get("information_ratio") is not None:
        lines.append(f"信息比率 {m['information_ratio']:.2f}（超额收益的稳定性，>0 说明稳定跑赢）")
    if m.get("n_days"):
        lines.append(f"回测区间 {m['n_days']} 个交易日")
    return lines
