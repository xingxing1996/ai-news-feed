"""冒烟测试：导入所有模块 + 合成数据下的因子/标签/评估/回测逻辑。

完整流水线（含 LightGBM）见 run_demo.sh；本测试覆盖纯 pandas 逻辑，便于在装好
核心依赖后快速回归。需要 stock-predict 环境（pandas/numpy/scipy）。
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
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
    assert "residual_return" in lab.columns
    assert lab["label"].dropna().isin([0, 1]).all()


def test_zone_label_is_market_local():
    """同一日期的不同市场不得相互决定强弱标签。"""
    from stock_predict.features.labels import zone_label

    idx = pd.MultiIndex.from_tuples(
        [("2024-01-02", c) for c in ("CN1", "CN2", "CN3", "US1", "US2", "US3")],
        names=["date", "code"],
    )
    # 全市场一起排会把所有 CN 判弱、所有 US 判强；按市场分组后各自都有强弱两端。
    signal = pd.Series([1.0, 2.0, 3.0, 101.0, 102.0, 103.0], index=idx)
    market = pd.Series(["cn", "cn", "cn", "us", "us", "us"], index=idx)
    label = zone_label(signal, market)
    assert label.loc[("2024-01-02", "CN1")] == 0
    assert label.loc[("2024-01-02", "CN3")] == 1
    assert label.loc[("2024-01-02", "US1")] == 0
    assert label.loc[("2024-01-02", "US3")] == 1


def test_daily_quality_gate_rejects_partial_or_stale_data():
    from stock_predict.config import AttrDict
    from stock_predict.data.loaders import validate_daily_quality

    universe = pd.DataFrame([
        {"code": "000001.SZ", "market": "cn"},
        {"code": "000002.SZ", "market": "cn"},
    ])
    daily = pd.DataFrame([{"date": "2026-08-07", "code": "000001.SZ"}])
    cfg = AttrDict({"data": {"synthetic": False, "quality_gate": {
        "enabled": True, "min_coverage": 0.95, "max_staleness_bdays": 2,
    }}})
    result = validate_daily_quality(daily, universe, cfg, reference_date="2026-08-13")
    assert not result["ok"]
    assert result["markets"]["cn"]["coverage"] == 0.5
    assert result["markets"]["cn"]["stale_bdays"] == 4


def test_tencent_symbol_mapping():
    from stock_predict.data.akshare_loader import _to_tx_symbol

    assert _to_tx_symbol("600519.SH") == "sh600519"
    assert _to_tx_symbol("000001.SZ") == "sz000001"
    assert _to_tx_symbol("688981.SH") == "sh688981"


def test_us_config_uses_label_ranker_not_cn_quality_signal(monkeypatch):
    """GitHub's yfinance job must not enter the A-share quality-only branch."""
    from stock_predict.config import load_settings, reset_settings
    from stock_predict.model.lgbm import active_targets

    monkeypatch.setenv("STOCK_PREDICT_CONFIG", "config/settings.us.yaml")
    reset_settings()
    cfg = load_settings()
    assert cfg.universe.markets == ["us", "hk", "kr"]
    assert cfg.backtest.ranking == "label"
    assert active_targets(cfg) == ["label"]
    assert cfg.data.quality_gate.enabled is True
    reset_settings()


def test_optional_tables_without_schema_are_safe_for_yfinance_runs():
    """US/HK ingestion may not create valuation/financial parquet files."""
    from stock_predict.features.build import _filter_codes

    no_table = pd.DataFrame()
    assert _filter_codes(no_table, {"AAPL"}).empty
    assert _filter_codes(no_table, {"AAPL"}, include=False).empty
    table = pd.DataFrame({"code": ["AAPL", "0700.HK"], "pe": [20.0, 18.0]})
    assert _filter_codes(table, {"AAPL"})["code"].tolist() == ["AAPL"]
    assert _filter_codes(table, {"AAPL"}, include=False)["code"].tolist() == ["0700.HK"]


def test_us_label_path_trains_lgbm_ranker_with_market_groups():
    """Offline regression test for the exact US/HK label training branch."""
    from stock_predict.model.lgbm import _train_one

    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    codes = [f"US{i}" for i in range(6)]
    index = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])
    frame = pd.DataFrame(index=index)
    frame["market"] = "us"
    frame["factor"] = np.tile(np.arange(len(codes), dtype=float), len(dates))
    frame["label"] = (frame["factor"] >= 3).astype(int)
    split = {
        "train": frame.loc[(slice(dates[0], dates[5]), slice(None)), :],
        "valid": frame.loc[(slice(dates[6], dates[7]), slice(None)), :],
        "_all": frame,
        "_seg": {"train_end": str(dates[5].date())},
    }
    params = {"n_estimators": 12, "learning_rate": 0.1, "num_leaves": 7,
              "min_child_samples": 1, "random_state": 7}
    _, rank, blob, n_train = _train_one(["factor"], params, split, "label")
    assert blob["kind"] == "ranker"
    assert n_train == 36
    ranked = pd.Series(rank, index=index)
    # Binary relevance naturally produces score ties; the important contract is
    # that ranks are computed separately for every market-date query.
    assert ranked.groupby(level="date").nunique().ge(2).all()
    assert ranked.groupby(level="date").min().gt(0).all()
    assert ranked.groupby(level="date").max().le(1.0).all()


def test_us_label_walkforward_runs_offline(monkeypatch):
    """The GitHub US/HK walk-forward branch must produce rank_label offline."""
    from stock_predict.backtest import walkforward as wf
    from stock_predict.config import AttrDict

    dates = pd.date_range("2023-01-02", periods=160, freq="B")
    codes = [f"US{i}" for i in range(15)]
    index = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])
    mat = pd.DataFrame(index=index).reset_index()
    mat["market"] = "us"
    mat["factor"] = np.tile(np.arange(len(codes), dtype=float), len(dates))
    mat["label"] = (mat["factor"] >= 8).astype(int)
    cfg = AttrDict({
        "feature": {"label_horizon": 20},
        "model": {"lightgbm": {"n_estimators": 8, "learning_rate": 0.1,
                                  "num_leaves": 7, "min_child_samples": 1, "random_state": 3},
                  "split": {"test_start": "2023-05-01"}},
        "backtest": {"ranking": "label"},
    })
    written = {}
    monkeypatch.setattr(wf, "get_settings", lambda: cfg)
    monkeypatch.setattr(wf, "read_parquet", lambda _name: mat.copy())
    monkeypatch.setattr(wf, "write_parquet", lambda frame, name: written.setdefault(name, frame.copy()))
    result = wf.walk_forward_oos(train_days=60, step=63)
    assert not result.empty
    assert {"date", "code", "market", "rank_label", "split"}.issubset(result.columns)
    assert result["rank_label"].between(0, 1).all()
    assert "predictions_oos" in written


def test_us_workflow_requires_validation_before_publish():
    workflow = (Path(__file__).resolve().parents[2] / ".github" / "workflows" / "train.yml").read_text()
    assert "Refusing to publish an unvalidated US/HK model" in workflow
    publish = workflow[workflow.index("- name: 提交 recommendations_us.json"):]
    assert "if: success()" in publish.split("run:", 1)[0]


def test_modelscope_sync_retries_and_never_hides_remote_errors():
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "sync_modelscope.sh").read_text()
    assert "for attempt in 1 2 3" in script
    assert "Unable to fetch ModelScope" in script
    assert "Unable to push to ModelScope" in script
    assert "fetch modelscope master || true" not in script
    for workflow_name in ("train.yml", "report.yml", "sync-modelscope.yml"):
        workflow = (root / ".github" / "workflows" / workflow_name).read_text()
        assert "bash scripts/sync_modelscope.sh" in workflow


def test_report_workflow_does_not_publish_after_a_failed_refresh():
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "report.yml").read_text()
    publish = workflow[workflow.index("- name: 提交 recommendations.json") :]
    assert "if: success()" in publish.split("run:", 1)[0]


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
