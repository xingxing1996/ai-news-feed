"""冒烟测试：导入所有模块 + 合成数据下的因子/标签/评估/回测逻辑。

完整流水线（含 LightGBM）见 run_demo.sh；本测试覆盖纯 pandas 逻辑，便于在装好
核心依赖后快速回归。需要 stock-predict 环境（pandas/numpy/scipy）。
"""
from __future__ import annotations

import importlib

import pandas as pd
import pytest


MODULES = [
    "stock_predict.config",
    "stock_predict.data.models",
    "stock_predict.data.warehouse",
    "stock_predict.data.universe",
    "stock_predict.data.akshare_loader",
    "stock_predict.data.yfinance_loader",
    "stock_predict.data.synthetic_loader",
    "stock_predict.data.loaders",
    "stock_predict.data.qlib_ingest",
    "stock_predict.features.alpha",
    "stock_predict.features.valuation",
    "stock_predict.features.quality",
    "stock_predict.features.industry",
    "stock_predict.features.labels",
    "stock_predict.features.build",
    "stock_predict.model.evaluate",
    "stock_predict.backtest.metrics",
    "stock_predict.report.explain",
    "stock_predict.news.sources",
    "stock_predict.news.llm_events",
]


def test_imports():
    for m in MODULES:
        importlib.import_module(m)


@pytest.fixture()
def synthetic_daily():
    from stock_predict.data.synthetic_loader import fetch_daily, fetch_valuation, fetch_financial

    codes = ["600519.SH", "000858.SZ", "AAPL", "0700.HK"]
    daily = pd.concat([fetch_daily(c, "2018-01-01", "2024-12-31") for c in codes], ignore_index=True)
    valuation = pd.concat([fetch_valuation(c, "2018-01-01", "2024-12-31") for c in codes], ignore_index=True)
    financial = pd.concat([fetch_financial(c) for c in codes], ignore_index=True)
    universe = pd.DataFrame(
        [
            {"code": "600519.SH", "name": "贵州茅台", "market": "cn", "industry": "食品饮料"},
            {"code": "000858.SZ", "name": "五粮液", "market": "cn", "industry": "食品饮料"},
            {"code": "AAPL", "name": "Apple", "market": "us", "industry": "科技"},
            {"code": "0700.HK", "name": "腾讯", "market": "hk", "industry": "互联网"},
        ]
    )
    return daily, valuation, financial, universe


def test_synthetic_loader_shapes(synthetic_daily):
    daily, valuation, financial, _ = synthetic_daily
    assert len(daily) > 1000
    assert set(["date", "code", "open", "high", "low", "close", "volume", "market_cap"]).issubset(daily.columns)
    assert {"date", "code", "pe", "pb"}.issubset(valuation.columns)
    assert {"code", "roe", "gross_margin", "cashflow"}.issubset(financial.columns)


def test_alpha_factors(synthetic_daily):
    from stock_predict.features.alpha import compute_alpha_factors

    daily, *_ = synthetic_daily
    feats = compute_alpha_factors(daily)
    assert not feats.empty
    assert feats.shape[1] >= 30  # pandas 兜底应有数十个价量因子


def test_valuation_factors(synthetic_daily):
    from stock_predict.features.valuation import compute_valuation_factors

    daily, valuation, financial, _ = synthetic_daily
    val = compute_valuation_factors(daily, valuation, financial)
    assert "pe_percentile" in val.columns or "fcf_yield" in val.columns


def test_labels(synthetic_daily):
    from stock_predict.features.labels import compute_labels

    daily, _, _, universe = synthetic_daily
    lab = compute_labels(daily, universe, horizon=20)
    assert "label" in lab.columns
    assert lab["label"].dropna().isin([0, 1]).all()


def test_evaluate():
    from stock_predict.model.evaluate import summarize

    prob = pd.Series([0.9, 0.1, 0.8, 0.2, 0.7] * 10)
    label = pd.Series([1, 0, 1, 0, 1] * 10)
    date = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"] * 10)
    m = summarize(prob, label, date)
    assert "rank_ic_mean" in m


def test_backtest_metrics():
    from stock_predict.backtest.metrics import compute

    port = pd.Series([0.001, -0.002, 0.003, 0.0] * 60)
    bench = pd.Series([0.0] * 240)
    m = compute(port, bench)
    assert "ann_return" in m and "max_drawdown" in m and "sharpe" in m


def test_llm_rule_fallback():
    from stock_predict.news.llm_events import events_to_score, _rule_extract

    # 直接测规则抽取（不触发真实 LLM 调用，保持测试快、不烧 API 配额）
    evs = _rule_extract("英伟达大幅增加 HBM 订单，需求强劲增长", ["NVDA"])
    assert len(evs) >= 1
    s = events_to_score(evs)
    assert 0.0 <= s <= 1.0


def test_explain():
    from stock_predict.report.explain import _lookup, explain_row

    assert _lookup("pe_percentile") is not None
    assert _lookup("ROC20") is not None
    # explain_row 简单冒烟
    import numpy as np

    section = pd.DataFrame({"pe_percentile": [0.1, 0.9], "ROC20": [0.05, -0.05]},
                           index=pd.MultiIndex.from_tuples([("d", "A"), ("d", "B")], names=["date", "code"]))
    row = section.loc[("d", "A")]
    row.name = ("d", "A")
    reasons, risks = explain_row(row, ["pe_percentile", "ROC20"], section, model=None)
    assert isinstance(reasons, list) and isinstance(risks, list)
