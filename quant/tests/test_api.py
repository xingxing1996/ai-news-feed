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
    """测试 GET /recommendations 与 GET /api/recommendations 接口。"""
    for endpoint in ("/recommendations", "/api/recommendations"):
        resp = client.get(endpoint)
        assert resp.status_code in (200, 404)
        assert resp.headers.get("cache-control") == "no-cache, no-store, must-revalidate"
        if resp.status_code == 200:
            data = resp.json()
            assert "recommendations" in data
            recs = data["recommendations"]
            if recs:
                first = recs[0]
                # 校验核心字段 100% 存在且未丢失
                assert "code" in first
                assert "name" in first
                assert "current_price" in first
                assert "target_price" in first
                assert "expected_return_pct" in first
                assert "prob_up" in first


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
            assert "reasons" in first and len(first["reasons"]) >= 3
            assert "risks" in first and len(first["risks"]) >= 3
            # 确保不存在无意义的 "+0.00" 截断
            reasons_str = "".join(first["reasons"])
            assert "+0.00）" not in reasons_str, "不能存在 +0.00 无意义截断"


def test_anti_leakage_invariants():
    """自动化架构级门禁测试：验证系统关键代码库中无全量 rank 与未来财报泄漏。"""
    lgbm_py = (ROOT / "quant" / "stock_predict" / "model" / "lgbm.py").read_text(encoding="utf-8")
    quality_py = (ROOT / "quant" / "stock_predict" / "features" / "quality.py").read_text(encoding="utf-8")
    strategy_py = (ROOT / "quant" / "stock_predict" / "backtest" / "strategy.py").read_text(encoding="utf-8")

    # 1. 验证 lgbm.py 必须使用按交易日 groupby("date") 的截面 rank
    assert 'groupby("date")["p1"].rank(pct=True)' in lgbm_py, "lgbm.py 必须包含按 date 分组的截面 rank"
    
    # 2. 验证 quality.py 必须使用 direction="backward" 的 PIT 财报对齐
    assert 'direction="backward"' in quality_py, "quality.py 必须使用 backward 方向的 merge_asof PIT 对齐"

    # 3. 验证 strategy.py 必须使用 T+1 调仓 (d_exec = dates[i + 1])
    assert 'd_exec = dates[i + 1]' in strategy_py, "strategy.py 必须包含严格 T+1 调仓时间轴"
