"""Top-N 持有策略 + 组合回测（设计文档第 9 节）。

策略：
- 在 test 段内，每隔 ``hold_days`` 个交易日调仓一次；
- 每次取当日预测概率 Top-N，等权持有到下次调仓；
- 每次调仓按换手率扣手续费 + 滑点。

输出：日度收益序列（组合 & 基准）、持仓记录、组合指标。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import get_settings
from ..data.universe import resolve_universe
from ..data.warehouse import read_parquet
from . import metrics

log = logging.getLogger(__name__)


def _weight_churn(prev: pd.Series, new: pd.Series) -> float:
    """两次持仓的权重换手率 = 0.5 * Σ|w_new - w_prev|（全集合并）。"""
    if prev.empty:
        return 1.0  # 首次建仓
    df = pd.concat([prev.rename("p"), new.rename("n")], axis=1).fillna(0.0)
    return float(0.5 * (df["n"] - df["p"]).abs().sum())


def _pivot_close(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.pivot(index="date", columns="code", values="close").sort_index()


def _pivot_prob(pred: pd.DataFrame) -> pd.DataFrame:
    # 兼容三概率格式：优先用「跑赢行业」prob_label，回退旧版 prob
    col = "prob_label" if "prob_label" in pred.columns else "prob"
    return pred.pivot(index="date", columns="code", values=col).sort_index()


def run_backtest() -> dict:
    cfg = get_settings()
    pred = read_parquet("predictions")
    daily = read_parquet("daily_price")
    if pred.empty or daily.empty:
        raise RuntimeError("predictions/daily_price 为空，请先 train。")

    top_n = int(cfg.backtest.top_n)
    hold = int(cfg.backtest.hold_days)
    comm = float(cfg.backtest.commission_bps) / 1e4
    slip = float(cfg.backtest.slippage_bps) / 1e4

    # 限定到 test 段
    seg = dict(cfg.model.split)
    test_start = pd.Timestamp(seg.get("test_start"))
    test_end = pd.Timestamp(seg.get("test_end")) if seg.get("test_end") else None
    prob = _pivot_prob(pred)
    close = _pivot_close(daily)
    # 统一为 datetime 索引，避免 str/Timestamp 比较报错
    prob.index = pd.to_datetime(prob.index)
    close.index = pd.to_datetime(close.index)
    # 对齐日期与代码
    common_codes = prob.columns.intersection(close.columns)
    dates = prob.index[(prob.index >= test_start) & (prob.index <= (test_end if test_end else prob.index.max()))]
    dates = dates.sort_values()
    if len(dates) == 0:
        raise RuntimeError("test 段内无预测日期，请检查切分配置。")

    ret = close.pct_change()
    # 基准：全池等权
    mkt_ret = ret.mean(axis=1)
    # 个股波动（用于风险加权）
    vol = ret.rolling(60, min_periods=20).std().fillna(0.02)
    # 行业映射（用于行业上限）
    from ..data.universe import resolve_universe
    ind_map = resolve_universe().set_index("code")["industry"].to_dict()

    weighting = cfg.backtest.get("weighting", "inv_vol")
    max_w = float(cfg.backtest.get("max_weight", 0.10))
    max_ind = float(cfg.backtest.get("max_industry", 0.40))
    impact_coef = float(cfg.backtest.get("impact_coef", 10.0))  # 平方根市场冲击系数(bps)

    port_dates: list = []
    port_ret: list[float] = []
    holdings_log: list[tuple] = []
    weights: pd.Series = pd.Series(dtype=float)
    prev_weights: pd.Series = pd.Series(dtype=float)
    turnover_sum = 0.0
    n_rebal = 0

    def _make_weights(codes: list[str], d) -> pd.Series:
        if not codes:
            return pd.Series(dtype=float)
        if weighting == "inv_vol":
            v = vol.loc[d, codes].clip(lower=1e-4) if d in vol.index else pd.Series(0.02, index=codes)
            w = 1.0 / v
        else:
            w = pd.Series(1.0, index=codes)
        w = (w / w.sum()).clip(upper=max_w)  # 单票上限
        # 行业上限
        ind_ser = pd.Series({c: ind_map.get(c, "其他") for c in w.index})
        for ind, codes_idx in ind_ser.groupby(ind_ser).groups.items():
            s = w.loc[codes_idx].sum()
            if s > max_ind and s > 0:
                w.loc[codes_idx] *= max_ind / s
        w = w / w.sum()  # 重归一
        return w

    for i, d in enumerate(dates):
        if i % hold == 0:
            row = prob.loc[d, common_codes].dropna()
            new_holdings = row.nlargest(top_n).index.tolist() if len(row) >= top_n else row.index.tolist()
            weights = _make_weights(new_holdings, d)
            # 换手率（按权重变动）+ 成本（佣金+滑点+平方根冲击）
            churn = _weight_churn(prev_weights, weights)
            impact = impact_coef * (churn ** 0.5) / 1e4  # √换手 的市场冲击
            cost = (comm + slip) * churn + impact
            prev_weights = weights
            holdings_log.append((str(d.date()), ",".join(weights.index)))
            n_rebal += 1
            turnover_sum += churn
        else:
            cost = 0.0

        if weights.empty:
            continue
        day_ret = ret.loc[d, weights.index] if d in ret.index else pd.Series(dtype=float)
        pr = float((weights * day_ret.fillna(0.0)).sum()) if not weights.empty else 0.0
        pr -= cost
        port_dates.append(d)
        port_ret.append(pr)

    port = pd.Series(port_ret, index=port_dates, name="portfolio")
    port.index.name = "date"
    bench = mkt_ret.reindex(port_dates).fillna(0.0).rename("benchmark")
    bench.index.name = "date"

    report = metrics.compute(port, bench)
    report.update(
        {
            "top_n": top_n,
            "hold_days": hold,
            "n_rebalance": n_rebal,
            "avg_turnover": float(turnover_sum / n_rebal) if n_rebal else 0.0,
            "start": str(pd.Timestamp(dates[0]).date()),
            "end": str(pd.Timestamp(dates[-1]).date()),
        }
    )
    # 落盘净值曲线
    eq = pd.DataFrame({"portfolio": (1 + port).cumprod(),
                       "benchmark": (1 + bench).cumprod()})
    eq.index.name = "date"
    from ..data.warehouse import write_parquet

    write_parquet(eq.reset_index(), "equity_curve")
    log.info("[backtest] %s", {k: v for k, v in report.items() if k in
             ("ann_return", "max_drawdown", "sharpe", "avg_turnover", "excess_ann")})
    return report
