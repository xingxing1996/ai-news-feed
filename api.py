"""结果查询 API（FastAPI）。

由容器常驻提供，读取 quant/data/output 下调度任务产出的 json/md：
  GET /health          健康检查
  GET /recommendations 最新推荐（recommendations_cn.json，A股+港股）
  GET /report          日报 markdown
  GET /backtest        回测指标
  GET /files           列出 output 目录文件

本地：uvicorn api:app --port 8000
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

OUT = Path(os.getenv("OUT_DIR", "quant/data/output"))
app = FastAPI(title="stock-predict API", description="量化推荐结果查询")


def _read_json(name: str):
    p = OUT / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"解析失败: {exc}"}


@app.get("/health")
def health():
    return {"status": "ok", "out_dir": str(OUT)}


@app.get("/recommendations")
def recommendations():
    """最新推荐（三概率 + 评分 + 理由/风险）。"""
    data = _read_json("recommendations_cn.json") or _read_json("recommendations.json")
    if data is None:
        return JSONResponse({"error": "暂无结果，请等待首次训练完成（看 /health 与日志）"}, status_code=404)
    return data


@app.get("/report")
def report():
    """日报 markdown。"""
    for name in ("recommendations_cn.md", "daily_report.md"):
        p = OUT / name
        if p.exists():
            return PlainTextResponse(p.read_text(encoding="utf-8"))
    return JSONResponse({"error": "暂无日报"}, status_code=404)


@app.get("/backtest")
def backtest():
    """回测指标。"""
    data = _read_json("backtest_metrics.txt") or _read_json("backtest_metrics_us.json")
    return data or {"error": "暂无回测结果"}


@app.get("/files")
def files():
    """列出 output 目录文件（调试用）。"""
    if not OUT.exists():
        return {"files": []}
    return {"files": [f.name for f in OUT.iterdir() if f.is_file()]}
