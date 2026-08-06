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
