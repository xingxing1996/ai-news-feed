"""合成数据生成器（离线兜底 / 测试 / 演示）。

目的：让 ingest → features → train → backtest → report 全链路在**无网络、无 API key**
的情况下也能端到端跑通，并产出有意义的日报。

数据注入了**可学习的弱结构**，使 PE 分位（均值回归）与动量（延续）对未来收益有预测力，
从而 Rank IC 非平凡、日报「理由」言之有物。这仅用于演示，不代表真实市场。

每只股票由 code 哈希定种子 → 可复现。
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def _seed(code: str) -> int:
    return int(hashlib.md5(code.encode("utf-8")).hexdigest(), 16) % (2**32)


def _trading_days(start: str, end: str) -> list[str]:
    dates = pd.bdate_range(start, end)
    return [d.strftime("%Y-%m-%d") for d in dates]


def _gen_series(code: str, start: str, end: str) -> pd.DataFrame:
    rng = np.random.default_rng(_seed(code))
    days = _trading_days(start, end)
    n = len(days)
    if n == 0:
        return _empty_daily()

    # 风格参数（每股不同）
    mu = rng.normal(0.10, 0.05)            # 年化漂移
    vol = rng.uniform(0.20, 0.45)          # 年化波动
    beta = rng.uniform(0.5, 1.4)           # 市场敏感度
    base = float(rng.uniform(20, 200))     # 起始价
    shares = float(10 ** rng.uniform(7, 10))

    # 市场因子（所有股票共享同一随机流时不同；这里每股自带一份，简化）
    mkt = rng.normal(0.08 / 252, 0.16 / np.sqrt(252), n)
    idio = rng.normal(0, vol / np.sqrt(252), n)

    # 盈利周期：~2 年一个周期，制造 PE 的均值回归信号
    cycle_len = int(rng.uniform(300, 520))
    t = np.arange(n)
    eps_cycle = 1 + 0.35 * np.sin(2 * np.pi * t / cycle_len)
    eps_growth = 1 + 0.02 * np.sin(2 * np.pi * t / cycle_len + rng.uniform(0, 2 * np.pi))

    # 收益 = 漂移 + 市场 + 个体 + 盈利周期的「价值反转」项
    value_revert = 0.15 / 252 * np.sin(2 * np.pi * t / cycle_len)  # 便宜时回升
    ret = mu / 252 + beta * mkt + idio + value_revert
    logprice = np.cumsum(ret) + np.log(base)
    close = np.exp(logprice)

    # EPS：基准 + 周期
    eps_base = base / float(rng.uniform(8, 30))  # 基准 PE 8~30
    eps = eps_base * eps_cycle * eps_growth
    pe = close / np.maximum(eps, 1e-6)
    pb = pe * float(rng.uniform(0.05, 0.4))      # 简化：PB 与 PE 挂钩
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    opn = close * (1 + rng.normal(0, 0.005, n))
    volume = np.exp(rng.normal(15, 1.5, n)) * shares / 1e7

    df = pd.DataFrame(
        {
            "date": days,
            "code": code,
            "open": open_round(opn),
            "high": open_round(high),
            "low": open_round(low),
            "close": open_round(close),
            "volume": volume,
            "market_cap": close * shares,
            "pe": pe,
            "pb": pb,
            "eps": eps,
        }
    )
    return df


def open_round(arr) -> np.ndarray:
    return np.round(arr, 2)


def fetch_daily(code: str, start: str, end: str) -> pd.DataFrame:
    df = _gen_series(code, start, end)
    if df.empty:
        return _empty_daily()
    return df[["date", "code", "open", "high", "low", "close", "volume", "market_cap"]].copy()


def fetch_valuation(code: str, start: str, end: str) -> pd.DataFrame:
    df = _gen_series(code, start, end)
    if df.empty:
        return _empty_valuation()
    return df[["date", "code", "pe", "pb"]].copy()


def fetch_financial(code: str, market: str = "cn") -> pd.DataFrame:
    """合成多期财报（够做 quality 因子，且 PIT 可对齐）。
    每期 report_period = 季度末，pub_date = 报告期 + 披露滞后（年报90天/季报45天），
    保证 merge_asof(backward) 只匹配已公开财报，与真实 loader 行为一致。"""
    rng = np.random.default_rng(_seed(code) + 1)
    today_dt = datetime.today()
    base_rev = float(rng.uniform(1e9, 5e10))
    base_margin = float(rng.uniform(0.15, 0.70))
    g = float(rng.uniform(0.72, 1.0))  # 期际温和增长因子
    last_day = {3: 31, 6: 30, 9: 30, 12: 31}
    cy, cq = today_dt.year, max(q for q in (3, 6, 9, 12) if q <= today_dt.month)
    periods = []
    for _ in range(8):
        periods.append((cy, cq))
        cq -= 3
        if cq <= 0:
            cq += 12
            cy -= 1
    periods = sorted(periods)
    rows = []
    for i, (yy, mm) in enumerate(periods):
        rp = datetime(yy, mm, last_day[mm])
        pub = rp + timedelta(days=90 if mm == 12 else 45)
        rev = base_rev * (g ** (len(periods) - 1 - i))
        profit = rev * base_margin
        rows.append({
            "code": code,
            "report_period": rp.strftime("%Y-%m-%d"),
            "pub_date": pub.strftime("%Y-%m-%d"),
            "revenue": rev,
            "profit": profit,
            "roe": float(rng.uniform(0.03, 0.30)),
            "gross_margin": base_margin,
            "cashflow": profit * float(rng.uniform(0.6, 1.8)),
            "pe": pd.NA,
            "pb": pd.NA,
        })
    return pd.DataFrame(rows)


def _empty_daily() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "code", "open", "high", "low", "close", "volume", "market_cap"])


def _empty_valuation() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "code", "pe", "pb"])


def _empty_fin() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["code", "report_period", "pub_date", "revenue", "profit", "roe",
                 "gross_margin", "cashflow", "pe", "pb"]
    )
