"""stock-predict 结果 API（FastAPI）+ APScheduler 进程内调度。

一个 uvicorn 进程同时提供：查询 API + 浏览器页面 + 后台定时（工作日17点训练 / 每2h刷新）。
适配 ModelScope 创空间（Docker 类型）：数据/模型走 /mnt/workspace，端口 7860。

端点：
  GET /                浏览器看板（推荐表格）
  GET /recommendations 最新推荐（JSON）
  GET /report          日报 markdown
  GET /backtest        回测指标
  GET /health          健康 + 调度状态
  GET /files           output 目录文件
  GET /docs            Swagger 文档
"""
from __future__ import annotations

import html
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

# ---- 环境在 import quant/调度 前 ----
ROOT = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = f"{ROOT}:{ROOT / 'quant'}"
os.environ.setdefault("STOCK_PREDICT_CONFIG", str(ROOT / "config" / "settings.modelspace.yaml"))
OUT = Path(os.environ.get("OUT_DIR", "/mnt/workspace/data/output"))
QUANT = ROOT / "quant"

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse  # noqa: E402

# apscheduler 容错：没装也能跑（仅失去进程内调度，外部 cron 仍可调度）
try:
    from apscheduler.schedulers.background import BackgroundScheduler  # noqa: E402
    from apscheduler.triggers.cron import CronTrigger  # noqa: E402
    from apscheduler.triggers.interval import IntervalTrigger  # noqa: E402
    _HAS_APS = True
except Exception:  # noqa: BLE001
    _HAS_APS = False

app = FastAPI(title="stock-predict API", description="A股+港股 量化推荐 + 定时调度")


# ---------- 调度 ----------
def _append_log(msg: str):
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "scheduler.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:  # noqa: BLE001
        pass


def _run_cli(cmd: str):
    """跑量化命令；输出直接流到控制台（ModelSpace 可见）+ 标记写入 scheduler.log。"""
    from datetime import datetime
    msg = f"[scheduler] === 开始 {cmd} ({datetime.now().strftime('%H:%M:%S')}) ==="
    print(msg, flush=True)
    _append_log(msg)
    try:
        # 不 capture：stdout/stderr 直接进容器控制台，出错能看到
        subprocess.run(
            [sys.executable, "-m", "stock_predict.cli", cmd],
            cwd=str(QUANT), env=dict(os.environ), timeout=3600,
        )
        msg = f"[scheduler] === {cmd} 完成 ==="
    except Exception as exc:  # noqa: BLE001
        msg = f"[scheduler] === {cmd} 失败: {exc} ==="
    print(msg, flush=True)
    _append_log(msg)


@app.on_event("startup")
def _start_scheduler():
    # 仅当装了 apscheduler 且未禁用时启动进程内调度（否则依赖外部 cron）
    if not _HAS_APS or os.getenv("USE_APSCHEDULER", "1") != "1":
        app.state.scheduler = None
        return
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(lambda: _run_cli("run"),
                  CronTrigger(hour=17, minute=0, day_of_week="mon-fri", timezone="Asia/Shanghai"),
                  id="train", replace_existing=True)
    sched.add_job(lambda: _run_cli("refresh"), IntervalTrigger(hours=2), id="refresh", replace_existing=True)
    sched.start()
    app.state.scheduler = sched
    threading.Thread(target=_run_cli, args=("run",), daemon=True).start()  # 启动先跑一次


# ---------- 读取 ----------
def _recs_path():
    for n in ("recommendations_cn.json", "recommendations.json"):
        p = OUT / n
        if p.exists():
            return p
    return None


def _read_json(name: str):
    p = OUT / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# ---------- API ----------
@app.get("/health")
def health():
    sched = getattr(app.state, "scheduler", None)
    jobs = [{"id": j.id, "next": str(j.next_run_time)} for j in sched.get_jobs()] if sched else []
    return {"status": "ok", "out_dir": str(OUT), "has_rec": _recs_path() is not None, "jobs": jobs}


@app.get("/recommendations")
def recommendations():
    data = _read_json("recommendations_cn.json") if (OUT / "recommendations_cn.json").exists() \
        else _read_json("recommendations.json")
    if data is None:
        return JSONResponse({"error": "暂无结果，首次训练进行中（约5–10分钟），看 /health"}, status_code=404)
    return data


@app.get("/report")
def report():
    for n in ("recommendations_cn.md", "daily_report.md"):
        p = OUT / n
        if p.exists():
            return PlainTextResponse(p.read_text(encoding="utf-8"))
    return JSONResponse({"error": "暂无日报"}, status_code=404)


@app.get("/backtest")
def backtest():
    return _read_json("backtest_metrics.txt") or {"error": "暂无回测"}


@app.get("/files")
def files():
    if not OUT.exists():
        return {"files": []}
    return {"files": [f.name for f in OUT.iterdir() if f.is_file()]}


@app.get("/log", response_class=PlainTextResponse)
def log_tail():
    """读取训练/调度日志尾部（排错用，看首次训练为什么没出结果）。"""
    for name in ("scheduler.log", "startup.log"):
        p = OUT / name
        if p.exists():
            return p.read_text(encoding="utf-8")[-6000:]
    return "_暂无日志（训练可能还没开始/在跑）_"


@app.get("/run")
def manual_run():
    """手动触发完整训练（后台）。"""
    threading.Thread(target=_run_cli, args=("run",), daemon=True).start()
    return {"status": "started", "msg": "训练已后台触发，看控制台日志或 /log"}


@app.get("/refresh")
def manual_refresh():
    """手动触发日报刷新（后台）。"""
    threading.Thread(target=_run_cli, args=("refresh",), daemon=True).start()
    return {"status": "started", "msg": "刷新已后台触发，看 /log"}


# ---------- 浏览器看板（根路径）----------
@app.get("/", response_class=HTMLResponse)
def dashboard():
    p = _recs_path()
    if not p:
        return HTMLResponse("<h2>stock-predict</h2><p>首次训练进行中，稍后刷新（看 <a href='/health'>/health</a>）。</p>")
    data = json.loads(p.read_text(encoding="utf-8"))
    recs = data.get("recommendations", [])
    cols = ["name", "code", "market", "industry", "prob_up", "prob_bench", "prob", "score", "confidence"]

    def cell(v):
        if isinstance(v, float):
            return f"{v*100:.0f}%"
        return html.escape(str(v))

    head = "".join(f"<th>{c}</th>" for c in cols)
    rows = ""
    for r in recs:
        rows += "<tr>" + "".join(f"<td>{cell(r.get(c, ''))}</td>" for c in cols) + "</tr>"

    return f"""<!doctype html><html><head><meta charset='utf-8'>
<title>A股+港股 量化推荐</title>
<style>body{{font-family:system-ui;margin:20px}} table{{border-collapse:collapse;font-size:14px}}
td,th{{border:1px solid #ddd;padding:5px 8px}} tr:nth-child(even){{background:#f7f7f7}} a{{color:#2563eb}}</style>
</head><body>
<h2>A股 + 港股 AI 量化推荐</h2>
<small>更新: {data.get('update_time','—')} | 三概率: 上涨 / 跑赢大盘 / 跑赢行业 | 非投资建议</small>
<p><a href='/recommendations'>JSON</a> | <a href='/report'>日报</a> | <a href='/backtest'>回测</a> | <a href='/docs'>API文档</a> | <a href='/health'>状态</a></p>
<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>
</body></html>"""
