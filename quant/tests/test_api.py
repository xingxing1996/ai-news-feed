"""FastAPI 核心 REST API 接口自动化单元测试集合 (pytest + TestClient)。

涵盖所有开放端点，保证每次修改代码后自动化跑单测，拦截 422 / 500 等路由与字段缺失回归。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# 确保 Python 能搜索到包根目录与 api.py
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant"))

from api import app

client = TestClient(app)


def test_dashboard_root():
    """测试 GET / 浏览器看板 (HTML 看板与 Cache-Control 响应头)。"""
    resp = client.get("/")
    assert resp.status_code in (200, 404)
    assert resp.headers.get("cache-control") == "no-cache, no-store, must-revalidate"
    if resp.status_code == 200:
        assert "<!doctype html>" in resp.text.lower() or "stock-predict" in resp.text


def test_recommendations_endpoint():
    """测试 GET /recommendations 与别名 /recommendations_us 推荐卡片接口。"""
    for endpoint in ("/recommendations", "/recommendations_us", "/api/recommendations_us"):
        resp = client.get(endpoint)
        assert resp.status_code in (200, 404, 503)
        assert resp.headers.get("cache-control") == "no-cache, no-store, must-revalidate"
        if resp.status_code == 200:
            data = resp.json()
            assert "recommendations" in data or "date" in data
            recs = data.get("recommendations", [])
            if recs:
                first = recs[0]
                # 校验核心字段 100% 存在且未丢失
                assert "code" in first
                assert "name" in first
                assert "current_price" in first
                assert "target_price" in first
                assert "expected_return_pct" in first
                assert "rank_up" in first


def test_report_endpoint():
    """测试 GET /report 与 GET /api/report 日报端点。"""
    for endpoint in ("/report", "/api/report"):
        resp = client.get(endpoint)
        assert resp.status_code in (200, 404)
        assert resp.headers.get("cache-control") == "no-cache, no-store, must-revalidate"


def test_health_endpoint():
    """测试 GET /health 与 GET /api/health 健康度检查端点。"""
    for endpoint in ("/health", "/api/health"):
        resp = client.get(endpoint)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "ok"
        assert "out_dir" in data
        assert "has_rec" in data


def test_backtest_endpoint():
    """测试 GET /backtest 接口。"""
    for endpoint in ("/backtest", "/api/backtest"):
        resp = client.get(endpoint)
        assert resp.status_code == 200


def test_files_endpoint():
    """测试 GET /files 接口。"""
    for endpoint in ("/files", "/api/files"):
        resp = client.get(endpoint)
        assert resp.status_code == 200
        data = resp.json()
        assert "files" in data
        assert isinstance(data["files"], list)


def test_log_tail_endpoint():
    """测试 GET /log 接口。"""
    for endpoint in ("/log", "/api/log"):
        resp = client.get(endpoint)
        assert resp.status_code == 200


def test_swagger_docs():
    """测试 Swagger /docs 接口。"""
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_refresh_endpoint():
    """测试 GET /refresh 接口。"""
    for endpoint in ("/refresh", "/api/refresh"):
        resp = client.get(endpoint)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "started"


def test_pe_dynamic_and_7_float_fields():
    """测试推荐卡片中是否 100% 包含 7 大纯 float 浮点数估值节点。"""
    resp = client.get("/recommendations")
    if resp.status_code == 200:
        data = resp.json()
        recs = data.get("recommendations", [])
        if recs:
            first = recs[0]
            # 校验 7 大 float 数值节点是否存在
            for key in ("pe", "raw_pe", "pe_dynamic", "pb", "raw_pb", "pe_percentile", "pb_percentile"):
                assert key in first, f"缺少关键估值节点 {key}"


def test_closed_loop_thesis():
    """测试卡片是否包含 5 维看多闭环 (bull_thesis) 与 3 维看空风控闭环 (bear_thesis)。"""
    resp = client.get("/recommendations")
    if resp.status_code == 200:
        data = resp.json()
        recs = data.get("recommendations", [])
        if recs:
            first = recs[0]
            assert "bull_thesis" in first, "缺少看多 5 维闭环 bull_thesis"
            assert "bear_thesis" in first, "缺少看空 3 维闭环 bear_thesis"
            assert isinstance(first["bull_thesis"], list) and len(first["bull_thesis"]) >= 3
            assert isinstance(first["bear_thesis"], list) and len(first["bear_thesis"]) >= 3


def test_detailed_reasons_and_risks():
    """测试利好 (reasons) 与利空 (risks) 是否包含高精度 SHAP 贡献度与特征真实数字明细。"""
    resp = client.get("/recommendations")
    if resp.status_code == 200:
        data = resp.json()
        recs = data.get("recommendations", [])
        if recs:
            first = recs[0]
            assert "reasons" in first and len(first["reasons"]) >= 2
            assert "risks" in first and len(first["risks"]) >= 2
            # 确保不存在无意义的 "+0.00" 截断
            reasons_str = "".join(first["reasons"])
            assert "+0.00）" not in reasons_str, "不能存在 +0.00 无意义截断"


def test_northlevel_is_point_in_time():
    """A1 门禁：north_level 必须用滚动秩，t 日值不得依赖未来数据。
    回归到全历史 g.rank(pct=True) 时，冷启动段不再为 NaN → 本断言失败。"""
    import numpy as np
    import pandas as pd
    from stock_predict.features import alt

    n = 400
    nb = pd.DataFrame({
        "code": ["000001.SZ"] * n,
        "date": pd.date_range("2020-01-01", periods=n).strftime("%Y-%m-%d"),
        "north_shares": np.arange(1, n + 1, dtype=float),  # 单调递增
    })
    full = alt.compute_northbound_factors(nb)
    # 冷启动期(<min_periods)必须为 NaN（全历史 rank 不会有 NaN → 据此识别未来函数）
    assert full["north_level"].iloc[:30].isna().all(), "north_level 冷启动段应 NaN，疑似用了全历史 rank(未来函数)"
    # PIT 性：截断未来后，过去点的 north_level 保持不变
    lvl_full = full.xs("000001.SZ", level="code")["north_level"]
    lvl_half = alt.compute_northbound_factors(nb.iloc[:201]).xs("000001.SZ", level="code")["north_level"]
    common = lvl_full.index.intersection(lvl_half.index)
    a, b = lvl_full.loc[common].dropna(), lvl_half.loc[common].dropna()
    shared = a.index.intersection(b.index)
    assert len(shared) > 0 and np.allclose(a.loc[shared].values, b.loc[shared].values), "north_level 依赖了未来数据"


def test_financial_pub_date_prevents_future_leak():
    """A2 门禁：pub_date 在未来的财报不得进入当前日特征（merge_asof backward PIT）。
    若 quality 退化为按报告期/今天对齐，roe=99 会泄漏进 2020 历史特征 → 本断言失败。"""
    import pandas as pd
    from stock_predict.features import quality

    dates = pd.bdate_range("2020-01-01", "2020-12-31")
    daily = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "code": "DUMMY"})
    # 一份"未来才发布"的财报：报告期 2020Q1，但 pub_date 远在未来
    fin = pd.DataFrame([{
        "code": "DUMMY", "report_period": "2020-03-31", "pub_date": "2025-12-31",
        "revenue": 1e9, "profit": 1e8, "roe": 99.0, "gross_margin": 0.5, "cashflow": 5e7,
    }])
    q = quality.compute_quality_factors(daily, fin)
    assert not q.empty, "quality 应返回与 daily 对齐的行"
    if "roe" in q.columns:
        assert not (q["roe"] == 99.0).any(), "未来财报(pub_date>交易日)泄漏进了历史特征！"


def test_production_configs_disable_synthetic():
    """门禁：所有 settings*.yaml 必须 synthetic:false，防止误开合成假数据进训练。"""
    import yaml
    cfg_dir = ROOT / "quant" / "config"
    yamls = list(cfg_dir.glob("settings*.yaml"))
    assert yamls, "未找到 settings yaml"
    for p in yamls:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        syn = (cfg.get("data") or {}).get("synthetic", False)
        assert syn is False, f"{p.name} 启用了 synthetic=true（合成假数据），生产训练禁止"


def test_loaders_do_not_stamp_today_as_pubdate():
    """A2 门禁：loader 不得把财报 pub_date 写死为今天（会让 PIT 对齐失效、历史特征全 NaN）。"""
    for fname in ("akshare_loader.py", "yfinance_loader.py", "synthetic_loader.py"):
        src = (ROOT / "quant" / "stock_predict" / "data" / fname).read_text(encoding="utf-8")
        assert '"pub_date": today' not in src and "'pub_date': today" not in src, (
            f"{fname} 仍把财报 pub_date 写死为 today，PIT 对齐会失效")
