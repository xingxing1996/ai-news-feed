"""Top-N 持有策略 + 组合回测（设计文档第 9 节）。

支持按不同信号给组合排序：
  - label  : 跑赢行业概率
  - abs    : 上涨概率
  - bench  : 跑赢大盘概率
  - blend  : 三者截面 rank 均值（综合分，默认）
可一次跑完四种并对比（run_backtest_compare）。
"""
from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd

from ..config import get_settings
from ..data.warehouse import read_parquet, write_parquet
from . import metrics

log = logging.getLogger(__name__)

_RANK_COLS = {"quality": "rank_quality", "residual": "rank_residual_return", "label": "rank_label", "abs": "rank_abs_label", "bench": "rank_bench_label"}
_LEGACY_PROB_COLS = {"label": "prob_label", "abs": "prob_abs_label", "bench": "prob_bench_label"}


def _weight_churn(prev: pd.Series, new: pd.Series) -> float:
    """两次持仓的权重换手率 = 0.5 * Σ|w_new - w_prev|。"""
    if prev.empty:
        return 1.0
    df = pd.concat([prev.rename("p"), new.rename("n")], axis=1).fillna(0.0)
    return float(0.5 * (df["n"] - df["p"]).abs().sum())


def _pivot_close(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.pivot(index="date", columns="code", values="close").sort_index()


def _ranking_signal(pred: pd.DataFrame, mode: str = "residual") -> pd.DataFrame:
    """返回 (date × code) 的排序信号，越大越买。"""
    if mode in _RANK_COLS and _RANK_COLS[mode] in pred.columns:
        return pred.pivot(index="date", columns="code", values=_RANK_COLS[mode]).sort_index()
    if mode in _LEGACY_PROB_COLS and _LEGACY_PROB_COLS[mode] in pred.columns:
        return pred.pivot(index="date", columns="code", values=_LEGACY_PROB_COLS[mode]).sort_index()
    if mode == "rank" and "rank" in pred.columns:  # walk-forward 兼容
        return pred.pivot(index="date", columns="code", values="rank").sort_index()
    if mode == "prob" and "prob" in pred.columns:  # 旧产物兼容
        return pred.pivot(index="date", columns="code", values="prob").sort_index()
    # blend：三个排序信号的截面 rank 均值
    cols = [c for c in _RANK_COLS.values() if c in pred.columns]
    if not cols:
        cols = [c for c in _LEGACY_PROB_COLS.values() if c in pred.columns]
    if not cols:
        col = "rank" if "rank" in pred.columns else ("prob" if "prob" in pred.columns else "rank_label")
        return pred.pivot(index="date", columns="code", values=col).sort_index()
    df = pred.copy()
    group_cols = ["date", "market"] if "market" in df.columns else ["date"]
    df["__blend"] = df.groupby(group_cols)[cols].rank(pct=True).mean(axis=1)
    return df.pivot(index="date", columns="code", values="__blend").sort_index()


def _benchmark_codes(benchmark, markets: set[str]) -> dict[str, str]:
    if isinstance(benchmark, Mapping):
        out = {market: str(benchmark[market]) for market in markets if benchmark.get(market)}
    else:
        out = {market: str(benchmark) for market in markets}
    missing = sorted(markets - set(out))
    if missing:
        raise RuntimeError(f"缺少市场基准配置：{', '.join(missing)}")
    return out


def _matched_benchmark_return(day_returns: pd.Series, weights: pd.Series,
                              market_by_code: dict[str, str], benchmark,
                              held_day_returns: pd.Series | None = None) -> float:
    """Return a market-weighted benchmark matching the portfolio's exposure."""
    market_weights_by_code = weights.index.to_series().map(market_by_code).fillna("other")
    market_weights = weights.groupby(market_weights_by_code).sum()
    benchmark = _benchmark_codes(benchmark, set(market_weights.index))
    result = 0.0
    for market, market_weight in market_weights.items():
        code = benchmark.get(market)
        value = day_returns.get(str(code))
        if value is None or pd.isna(value):
            held_codes = market_weights_by_code[market_weights_by_code.eq(market)].index
            market_traded = held_day_returns is not None and held_day_returns.reindex(held_codes).notna().any()
            if market_traded:
                raise RuntimeError(f"基准 {code} 在回测日缺少行情，不能计算可信超额收益。")
            # Different market calendars are expected in a multi-market book:
            # if this market did not trade, both its portfolio and benchmark
            # return are zero for the day.
            continue
        result += float(market_weight) * float(value)
    return result


def _run_core(signal: pd.DataFrame, daily: pd.DataFrame, cfg) -> dict:
    """给定排序信号，跑严格无未来函数的 T+1 真实调仓持有组合回测，返回指标 dict。"""
    top_n = int(cfg.backtest.top_n)
    hold = int(cfg.backtest.hold_days)
    comm = float(cfg.backtest.commission_bps) / 1e4
    slip = float(cfg.backtest.slippage_bps) / 1e4
    impact_coef = float(cfg.backtest.get("impact_coef", 10.0))
    weighting = cfg.backtest.get("weighting", "inv_vol")
    max_w = float(cfg.backtest.get("max_weight", 0.10))
    max_ind = float(cfg.backtest.get("max_industry", 0.40))

    seg = dict(cfg.model.split)
    test_start = pd.Timestamp(seg.get("test_start"))
    test_end = pd.Timestamp(seg.get("test_end")) if seg.get("test_end") else None

    close_all = _pivot_close(daily)
    signal.index = pd.to_datetime(signal.index)
    close_all.index = pd.to_datetime(close_all.index)
    common_codes = signal.columns.intersection(close_all.columns)
    dates = signal.index[(signal.index >= test_start)
                         & (signal.index <= (test_end if test_end else signal.index.max()))].sort_values()

    # 1. 计算个股 T+1 真实的隔日收益率 (ret_t1 = close[t] / close[t-1] - 1)
    # 组合收益只使用本次预测涉及的股票；基准从全量行情中按配置读取。
    close = close_all.loc[:, common_codes]
    # The raw table is a union of CN/HK/US trading calendars. Forward-fill
    # each close before returns so a foreign holiday is a zero-return day,
    # rather than an NaN that erases the next real trading day's return.
    ret = close.ffill().pct_change(fill_method=None)
    benchmark_ret = close_all.ffill().pct_change(fill_method=None)
    # Old mixed-market snapshots can contain a date from another exchange
    # (including weekends) even though no current-universe name traded. Those
    # dates cannot be a signal or T+1 execution date for this portfolio.
    actual_trade_dates = close.notna().any(axis=1)
    dates = dates[dates.isin(close.index)]
    dates = dates[actual_trade_dates.reindex(dates).fillna(False).to_numpy()]
    vol = ret.rolling(60, min_periods=20).std().fillna(0.02)
    # 成交量透视：用于停牌/零成交过滤（选股剔除信号日零成交的票，模拟"买不进"）
    volume_pivot = daily.pivot(index="date", columns="code", values="volume").sort_index() if "volume" in daily.columns else pd.DataFrame()
    if not volume_pivot.empty:
        volume_pivot.index = pd.to_datetime(volume_pivot.index)
    from ..data.universe import resolve_universe
    _uni = resolve_universe()
    ind_map = _uni.set_index("code")["industry"].to_dict()
    mkt_map = _uni.set_index("code")["market"].to_dict()
    benchmark_codes = _benchmark_codes(
        cfg.backtest.get("benchmark"), {mkt_map.get(code, "other") for code in common_codes}
    )
    missing_prices = [code for code in benchmark_codes.values() if code not in close_all.columns]
    if missing_prices:
        raise RuntimeError(f"回测基准缺少行情列：{', '.join(sorted(set(missing_prices)))}")
    # A stale benchmark must not extend the reported OOS period. Restrict the
    # end date to the last bar shared by every configured market benchmark.
    benchmark_last_dates = [close_all[code].last_valid_index() for code in benchmark_codes.values()]
    if any(date is None for date in benchmark_last_dates):
        raise RuntimeError("回测基准没有有效收盘价，不能计算可信超额收益。")
    last_benchmark_date = min(benchmark_last_dates)
    dates = dates[dates <= last_benchmark_date]
    if len(dates) < 2:
        raise RuntimeError("基准可用区间内预测日期不足 2 天，无法进行 T+1 严格无未来函数回测。")
    # 印花税按市场区分（卖方bp）：A 股 10、港股 13、美股/韩股 0；可用 backtest.stamp_duty_bps 覆盖
    _stamp_cfg = cfg.backtest.get("stamp_duty_bps") or {"cn": 10, "hk": 13, "us": 0, "kr": 0, "other": 0}

    def _stamp_frac(code: str) -> float:
        if isinstance(_stamp_cfg, dict):
            return float(_stamp_cfg.get(mkt_map.get(code, "other"), _stamp_cfg.get("other", 0))) / 1e4
        return float(_stamp_cfg or 0) / 1e4  # 单值兜底

    def _make_weights(codes, d):
        if not codes:
            return pd.Series(dtype=float)
        if weighting == "inv_vol":
            v = vol.loc[d, codes].clip(lower=1e-4) if d in vol.index else pd.Series(0.02, index=codes)
            w = 1.0 / v
        else:
            w = pd.Series(1.0, index=codes)
        w = (w / w.sum()).clip(upper=max_w)
        ind_ser = pd.Series({c: ind_map.get(c, "其他") for c in w.index})
        for _, codes_idx in ind_ser.groupby(ind_ser).groups.items():
            s = w.loc[codes_idx].sum()
            if s > max_ind and s > 0:
                w.loc[codes_idx] *= max_ind / s
        return w / w.sum()

    port_dates, port_ret, bench_ret, prev_weights, current_weights = [], [], [], pd.Series(dtype=float), pd.Series(dtype=float)
    turnover_sum = n_rebal = 0
    next_cost = 0.0

    # 2. 严格 T+1 调仓循环：d_signal 日产生收盘信号，d_exec 日才执行调仓并承受 d_exec 的真实收益与交易摩擦
    for i in range(len(dates) - 1):
        d_signal = dates[i]      # T 日：收盘后计算模型预测信号
        d_exec = dates[i + 1]    # T+1 日：实际调仓并承担持仓收益

        # 是否触发调仓
        if i % hold == 0:
            row = signal.loc[d_signal, common_codes].dropna()
            # 真实性：剔除信号日停牌/零成交的票（买不进）
            if d_signal in volume_pivot.index:
                vols = volume_pivot.loc[d_signal, row.index]
                row = row[vols.fillna(0).gt(0)]
            new_holdings = row.nlargest(top_n).index.tolist() if len(row) >= top_n else row.index.tolist()
            new_weights = _make_weights(new_holdings, d_signal)
            churn = _weight_churn(prev_weights, new_weights)
            # 成本：佣金 + 滑点 + 市场感知印花税(按新持仓市场构成加权) + √换手冲击
            stamp = float((new_weights.index.to_series().map(_stamp_frac) * new_weights).sum()) if not new_weights.empty else 0.0
            next_cost = (comm + slip + stamp) * churn + impact_coef * (churn ** 0.5) / 1e4
            current_weights = new_weights
            prev_weights = current_weights
            n_rebal += 1
            turnover_sum += churn
        else:
            next_cost = 0.0

        if current_weights.empty:
            continue

        # 计算 T+1 日调仓后的真实盘中收益率
        day_ret = ret.loc[d_exec, current_weights.index] if d_exec in ret.index else pd.Series(dtype=float)
        pr = float((current_weights * day_ret.fillna(0.0)).sum())
        
        port_dates.append(d_exec)
        port_ret.append(pr - next_cost)
        bench_ret.append(_matched_benchmark_return(
            benchmark_ret.loc[d_exec], current_weights, mkt_map, cfg.backtest.get("benchmark"),
            held_day_returns=day_ret,
        ))

    port = pd.Series(port_ret, index=port_dates, name="portfolio")
    port.index.name = "date"
    bench = pd.Series(bench_ret, index=port_dates, name="benchmark")
    bench.index.name = "date"
    report = metrics.compute(port, bench)
    report.update({"top_n": top_n, "hold_days": hold, "n_rebalance": n_rebal,
                   "avg_turnover": float(turnover_sum / n_rebal) if n_rebal else 0.0,
                   "benchmark": dict(cfg.backtest.get("benchmark", {})) if isinstance(cfg.backtest.get("benchmark"), Mapping)
                   else cfg.backtest.get("benchmark", "universe_equal_weight"),
                   "start": str(pd.Timestamp(dates[1]).date()), "end": str(pd.Timestamp(dates[-1]).date())})
    return report, port, bench


def run_backtest(ranking: str | None = None, pred_name: str = "predictions") -> dict:
    """按指定/默认排序信号回测。ranking: residual|label|abs|bench|blend。
    pred_name: 读哪份预测(默认 predictions；walk-forward OOS 用 predictions_oos)。"""
    cfg = get_settings()
    pred = read_parquet(pred_name)
    daily = read_parquet("daily_price")
    if pred.empty or daily.empty:
        raise RuntimeError("predictions/daily_price 为空，请先 train。")
    mode = ranking or cfg.backtest.get("ranking", "residual")
    report, port, bench = _run_core(_ranking_signal(pred, mode), daily, cfg)
    report["ranking"] = mode
    eq = pd.DataFrame({"portfolio": (1 + port).cumprod(), "benchmark": (1 + bench).cumprod()})
    eq.index.name = "date"
    write_parquet(eq.reset_index(), "equity_curve")
    log.info("[backtest] ranking=%s %s", mode,
             {k: v for k, v in report.items() if k in ("ann_return", "sharpe", "max_drawdown", "excess_ann")})
    return report


def run_backtest_compare() -> dict:
    """一次跑 label/abs/bench/blend 四种排序并对比。"""
    cfg = get_settings()
    pred = read_parquet("predictions")
    daily = read_parquet("daily_price")
    if pred.empty or daily.empty:
        raise RuntimeError("predictions/daily_price 为空，请先 train。")
    compare = {}
    for mode in ("residual", "label", "abs", "bench", "blend"):
        try:
            rep, _, _ = _run_core(_ranking_signal(pred, mode), daily, cfg)
            compare[mode] = {k: rep.get(k) for k in ("ann_return", "sharpe", "max_drawdown", "excess_ann", "avg_turnover")}
        except Exception as exc:  # noqa: BLE001
            compare[mode] = {"error": str(exc)}
    log.info("[backtest-compare] %s", compare)
    return compare
