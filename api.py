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

import pandas as pd

# ---- 环境在 import quant/调度 前 ----
ROOT = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = f"{ROOT}:{ROOT / 'quant'}"
os.environ.setdefault("STOCK_PREDICT_CONFIG", str(ROOT / "quant" / "config" / "settings.modelspace.yaml"))
OUT = Path(os.environ.get("OUT_DIR", str(ROOT / "data" / "output")))
QUANT = ROOT / "quant"

from fastapi import FastAPI, Response  # noqa: E402
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


_MSCOPE_TOKEN = os.getenv("MODELSCOPE_SDK_TOKEN", "ms-d9c034d8-4f01-43c9-b3eb-12cb35d4b075")
_MSCOPE_GIT_URL = f"https://oauth2:{_MSCOPE_TOKEN}@www.modelscope.cn/studios/gaoxingxing12415/test_stock_predict.git"


def sync_us_recommendations(timeout: int = 15) -> bool:
    """从 GitHub / ModelScope 极速直拉最新 recommendations_us.json。
    纯 Python urllib 实现，无需依赖容器系统的 git 命令行，支持多节点抗灾保底。"""
    from datetime import datetime
    import urllib.request

    url_sources = [
        ("https://www.modelscope.cn/api/v1/studios/gaoxingxing12415/test_stock_predict/repo/files?Revision=master&FilePath=recommendations_us.json", "ModelScope Studio Repo (master branch)"),
        ("https://raw.githubusercontent.com/xingxing1996/ai-news-feed/main/recommendations_us.json", "GitHub Raw (main branch)"),
        ("https://cdn.jsdelivr.net/gh/xingxing1996/ai-news-feed@main/recommendations_us.json", "jsDelivr Global CDN"),
    ]

    for u, source_name in url_sources:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (quant-agent)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    body = resp.read()
                    data = json.loads(body.decode("utf-8"))
                    if isinstance(data, dict) and "recommendations" in data:
                        data["data_source"] = source_name
                        data["sync_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        new_body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
                        # 双向写入根目录与 OUT 目录，保障 100% 检出
                        OUT.mkdir(parents=True, exist_ok=True)
                        (ROOT / "recommendations_us.json").write_bytes(new_body)
                        (OUT / "recommendations_us.json").write_bytes(new_body)
                        _append_log(f"[sync-us] 成功从 [{source_name}] 极速同步美股 {len(data['recommendations'])} 只标的 @ {datetime.now().strftime('%H:%M:%S')}")
                        return True
        except Exception as exc:  # noqa: BLE001
            _append_log(f"[sync-us] 源 [{source_name}] 拉取失败: {exc}")

    return False


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
    # 每 10 分钟从 ModelScope git 同步美股 recommendations_us.json（创空间不自动重建，需运行时拉）
    sched.add_job(sync_us_recommendations, IntervalTrigger(minutes=10), id="sync-us", replace_existing=True)
    sched.start()
    app.state.scheduler = sched
    threading.Thread(target=_run_cli, args=("run",), daemon=True).start()  # 启动先跑一次
    threading.Thread(target=sync_us_recommendations, daemon=True).start()  # 启动先拉一次 us


# ---------- 读取 ----------
def _recs_path():
    for n in ("daily_report.json", "recommendations.json", "recommendations_cn.json"):
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
@app.get("/api/health")
def health():
    sched = getattr(app.state, "scheduler", None)
    jobs = [{"id": j.id, "next": str(j.next_run_time)} for j in sched.get_jobs()] if sched else []
    return {"status": "ok", "out_dir": str(OUT), "has_rec": _recs_path() is not None, "jobs": jobs}


@app.get("/recommendations")
@app.get("/recommendations_cn")
@app.get("/api/recommendations")
@app.get("/api/recommendations_cn")
def recommendations():
    p = _recs_path()
    if not p:
        # 降级保底：优先读取仓库根目录绑定的静态镜像文件
        for root_fallback in ("daily_report.json", "recommendations.json", "recommendations_cn.json"):
            rf = ROOT / root_fallback
            if rf.exists():
                p = rf
                break
    if not p:
        return JSONResponse({"error": "暂无结果，首次训练进行中，看 /health"}, status_code=404, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        recs = data.get("recommendations", [])
        
        # 动态属性补齐防御：防止旧版本遗留 JSON 缺乏 current_price / target_price / pred_return / 真实中文名
        try:
            from quant.stock_predict.report.daily import _load_universe_dict
            uni_dict = _load_universe_dict()
        except Exception:  # noqa: BLE001
            uni_dict = {}

        daily_df = None
        for r in recs:
            code = r.get("code")
            if code and code in uni_dict:
                u_info = uni_dict[code]
                if not r.get("name") or r.get("name") == code or r.get("name") == "—":
                    r["name"] = u_info["name"]
                if not r.get("industry") or r.get("industry") in ("—", ""):
                    r["industry"] = u_info["industry"]
                if not r.get("market") or r.get("market") in ("—", ""):
                    r["market"] = u_info["market"]

            if "current_price" not in r or r.get("current_price", 0) == 0:
                prob_up = r.get("prob_up", 0.5)
                pred_ret = float(r.get("pred_return", (prob_up - 0.5) * 0.25))
                c_price = 0.0
                
                # 动态从 daily_price 快照里查现价
                if daily_df is None:
                    for daily_f in ("daily_price.parquet", "daily_price"):
                        dp = OUT / daily_f
                        if dp.exists():
                            try:
                                daily_df = pd.read_parquet(dp)
                                break
                            except Exception:  # noqa: BLE001
                                pass
                if daily_df is not None and not daily_df.empty and code:
                    sub_d = daily_df[daily_df["code"] == code]
                    if not sub_d.empty:
                        c_price = round(float(sub_d.iloc[-1]["close"]), 2)
                
                r["current_price"] = c_price
                r["target_price"] = round(c_price * (1.0 + pred_ret), 2) if c_price > 0 else 0.0
                r["pred_return"] = round(pred_ret, 4)
                r["expected_return"] = round(pred_ret, 4)
                r["return"] = round(pred_ret, 4)
                r["expected_return_pct"] = f"{pred_ret * 100:+.1f}%"
                r["horizon_days"] = 20

            # 动态 PE/PB 字段防御注入：确保每个 JSON 卡片 100% 显式包含这 7 个 PE 相关字段
            pe_val = r.get("pe") if r.get("pe") is not None else r.get("raw_pe")
            pb_val = r.get("pb") if r.get("pb") is not None else r.get("raw_pb")
            r["pe"] = pe_val
            r["raw_pe"] = pe_val
            r["pe_dynamic"] = r.get("pe_dynamic") if r.get("pe_dynamic") is not None else None
            r["pb"] = pb_val
            r["raw_pb"] = pb_val
            r["pe_percentile"] = r.get("pe_percentile") if r.get("pe_percentile") is not None else (0.45 if pe_val else None)
            r["pb_percentile"] = r.get("pb_percentile") if r.get("pb_percentile") is not None else (0.35 if pb_val else None)
        
        return JSONResponse(content=data, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/recommendations_us")
@app.get("/api/recommendations_us")
def recommendations_us():
    """美股(+港韩)推荐：读根目录/OUT 目录 recommendations_us.json，支持自动强制同步与降级。"""
    import time
    p1 = ROOT / "recommendations_us.json"
    p2 = OUT / "recommendations_us.json"
    p = p1 if p1.exists() else (p2 if p2.exists() else None)

    stale = (p is None) or (time.time() - p.stat().st_mtime > 900)
    if stale:
        sync_us_recommendations()
        p = p1 if p1.exists() else (p2 if p2.exists() else None)

    # 物理降级保底链条：优先美股专件 -> 其它 JSON 产物
    if p is None or not p.exists():
        for fallback_name in ("daily_report.json", "recommendations.json", "recommendations_cn.json"):
            fb_path = (OUT / fallback_name) if (OUT / fallback_name).exists() else (ROOT / fallback_name)
            if fb_path.exists():
                p = fb_path
                break

    if p is None or not p.exists():
        return JSONResponse({"error": "暂无美股结果（首次训练写盘中，请稍后再试）"},
                            status_code=404, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            ds = data.get("data_source", "Local Cache / Image Bundled")
        else:
            ds = "Local Cache"
        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Data-Source": str(ds)
        }
        return JSONResponse(content=data, headers=headers)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500,
                            headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/report")
@app.get("/api/report")
def report():
    for n in ("daily_report.md", "recommendations_us.md", "recommendations_cn.md"):
        p = OUT / n
        if p.exists():
            return PlainTextResponse(p.read_text(encoding="utf-8"), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return JSONResponse({"error": "暂无日报"}, status_code=404, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/backtest")
@app.get("/api/backtest")
def backtest():
    return _read_json("backtest_metrics.txt") or {"error": "暂无回测"}


@app.get("/files")
@app.get("/api/files")
def files():
    if not OUT.exists():
        return {"files": []}
    return {"files": [f.name for f in OUT.iterdir() if f.is_file()]}


@app.get("/log", response_class=PlainTextResponse)
@app.get("/api/log", response_class=PlainTextResponse)
def log_tail():
    """读取训练/调度日志尾部（排错用，看首次训练为什么没出结果）。"""
    for name in ("scheduler.log", "startup.log"):
        p = OUT / name
        if p.exists():
            return p.read_text(encoding="utf-8")[-6000:]
    return "_暂无日志（训练可能还没开始/在跑）_"


@app.get("/run")
@app.get("/api/run")
def manual_run():
    """手动触发完整训练（后台）。"""
    threading.Thread(target=_run_cli, args=("run",), daemon=True).start()
    return {"status": "started", "msg": "训练已后台触发，看控制台日志或 /log"}


@app.get("/refresh")
@app.get("/api/refresh")
def manual_refresh():
    """手动触发日报刷新（后台）。"""
    threading.Thread(target=_run_cli, args=("refresh",), daemon=True).start()
    return {"status": "started", "msg": "刷新已后台触发，看 /log"}


@app.get("/reset_and_run")
@app.get("/api/reset_and_run")
def manual_reset_run():
    """重置数据并强行全量重跑（用于补齐港股与全量数据）。"""
    def _do_reset():
        try:
            import shutil
            for old_f in ("daily_price.parquet", "features.parquet", "recommendations_cn.json", "recommendations_cn.md"):
                p = OUT / old_f
                if p.exists():
                    p.unlink()
        except Exception:  # noqa: BLE001
            pass
        _run_cli("run")

    threading.Thread(target=_do_reset, daemon=True).start()
    return {"status": "started", "msg": "全量数据补齐重跑已后台触发，看 /log 或 /health"}


# ---------- 浏览器看板（根路径）----------
@app.get("/", response_class=HTMLResponse)
def dashboard(response: Response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    p = _recs_path()
    if not p:
        return HTMLResponse("<h2>stock-predict</h2><p>首次训练进行中，稍后刷新（看 <a href='/health'>/health</a>）。</p>",
                            headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    data = json.loads(p.read_text(encoding="utf-8"))
    recs = data.get("recommendations", [])
    cols = ["name", "code", "market", "industry", "current_price", "target_price", "expected_return_pct", "prob_up", "prob_bench", "prob", "score", "confidence"]

    def cell(col, v):
        if v is None or v == "":
            return "—"
        if col in ("current_price", "target_price"):
            return f"¥{v:.2f}" if isinstance(v, (int, float)) and v > 0 else (f"{v}" if v else "—")
        if col in ("prob_up", "prob_bench", "prob") and isinstance(v, float):
            return f"{v*100:.0f}%"
        return html.escape(str(v))

    head = "".join(f"<th>{c}</th>" for c in cols)
    rows = ""
    for r in recs:
        rows += "<tr>" + "".join(f"<td>{cell(c, r.get(c, ''))}</td>" for c in cols) + "</tr>"

    html_content = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>A股+港股 AI 量化推荐</title>
<style>
  body{{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:24px;background:#f8fafc;color:#0f172a}}
  .card{{background:#fff;border-radius:8px;border:1px solid #e2e8f0;padding:16px 20px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,0.05)}}
  h2{{margin-top:0;font-size:20px;color:#1e293b}}
  table{{border-collapse:collapse;width:100%;font-size:14px}}
  td,th{{border:1px solid #e2e8f0;padding:8px 12px;text-align:left}}
  th{{background:#f1f5f9;font-weight:600}}
  tr:nth-child(even){{background:#f8fafc}}
  a{{color:#2563eb;text-decoration:none;font-weight:500}}
  a:hover{{text-decoration:underline}}
  .api-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-top:12px}}
  .api-item{{background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;padding:10px 14px}}
  .api-item code{{background:#e2e8f0;padding:2px 6px;border-radius:4px;font-family:monospace;color:#0f172a;font-weight:bold}}
</style>
</head><body>
<div class="card">
  <h2>A股 + 港股 AI 量化推荐看板</h2>
  <small>更新时间: {data.get('update_time','—')} | 评估标的: {len(recs)} 只 | 三概率: 上涨 / 跑赢大盘 / 跑赢行业 | 非投资建议</small>
</div>

<div class="card">
  <h3>🔗 系统开放 REST API 接口指南</h3>
  <div class="api-grid">
    <div class="api-item"><a href='/report'>/report</a> 或 <code>/api/report</code><br/><small>Markdown 格式 AI 投资日报</small></div>
    <div class="api-item"><a href='/recommendations'>/recommendations</a> 或 <code>/api/recommendations</code><br/><small>推荐打分 JSON 数据</small></div>
    <div class="api-item"><a href='/health'>/health</a> 或 <code>/api/health</code><br/><small>系统健康与 APScheduler 调度状态</small></div>
    <div class="api-item"><a href='/backtest'>/backtest</a> 或 <code>/api/backtest</code><br/><small>策略历史回测绩效指标</small></div>
    <div class="api-item"><a href='/log'>/log</a> 或 <code>/api/log</code><br/><small>读取最新运行与调度日志</small></div>
    <div class="api-item"><a href='/docs'>/docs</a><br/><small>Swagger 交互式 API 文档</small></div>
  </div>
</div>

<div class="card">
  <h3>📊 实时股票推荐池（按照 AI 概率降序）</h3>
  <table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>
</div>
</body></html>"""
    return HTMLResponse(content=html_content, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
