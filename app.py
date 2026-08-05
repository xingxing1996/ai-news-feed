"""ModelScope 创空间入口：A股+港股 量化看板 + APScheduler 进程内定时。

- 创空间只持久化 /mnt/workspace：数据/模型/日报全放那（重启不丢）。
- APScheduler（进程内）替代 cron：工作日 17:00 训练 + 每 2h 刷新（复用模型）。
- Gradio 看板：推荐表格 / 日报 / 回测 / 手动触发 / 调度状态。

部署：创空间 Docker 类型，用 Dockerfile.modelspace；或装 requirements-modelspace.txt 后跑 app.py。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

# ---- 环境必须在 import quant 之前设好 ----
ROOT = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = f"{ROOT}:{ROOT / 'quant'}"
os.environ.setdefault("STOCK_PREDICT_CONFIG", str(ROOT / "config" / "settings.modelspace.yaml"))
OUT = Path(os.environ.get("OUT_DIR", "/mnt/workspace/data/output"))
QUANT = ROOT / "quant"

import pandas as pd  # noqa: E402

try:
    import gradio as gr
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"缺少 gradio/apscheduler，请 pip install -r requirements-modelspace.txt：{exc}")


# ---------- 调度任务（子进程调 CLI，复用已测好的流程） ----------
def _run_cli(cmd: str) -> str:
    env = dict(os.environ)
    try:
        r = subprocess.run(
            [sys.executable, "-m", "stock_predict.cli", cmd],
            cwd=str(QUANT), env=env, capture_output=True, text=True, timeout=3600,
        )
        return (r.stdout or "") + (r.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return f"运行失败: {exc}"


def job_train():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scheduler.log").write_text(_run_cli("run"), encoding="utf-8")


def job_refresh():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scheduler.log").write_text(_run_cli("refresh"), encoding="utf-8")


# ---------- 启动调度器 ----------
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(
    job_train, CronTrigger(hour=17, minute=0, day_of_week="mon-fri", timezone="Asia/Shanghai"),
    id="train", replace_existing=True,
)
scheduler.add_job(job_refresh, IntervalTrigger(hours=2), id="refresh", replace_existing=True)
scheduler.start()
# 启动后台先跑一次训练（首次部署/重启后尽快有结果）
threading.Thread(target=job_train, daemon=True).start()


# ---------- 数据读取 ----------
def _recs_path():
    for n in ("recommendations_cn.json", "recommendations.json"):
        p = OUT / n
        if p.exists():
            return p
    return None


def recs_df():
    p = _recs_path()
    if not p:
        return pd.DataFrame(), "暂无结果，首次训练进行中（约 5–10 分钟）…"
    data = json.loads(p.read_text(encoding="utf-8"))
    df = pd.DataFrame(data.get("recommendations", []))
    for c in ("prob_up", "prob_bench", "prob"):
        if c in df:
            df[c] = (df[c].astype(float) * 100).round(0).astype(int).astype(str) + "%"
    cols = [c for c in
            ("code", "name", "market", "industry", "prob_up", "prob_bench", "prob", "score", "confidence")
            if c in df.columns]
    return df[cols] if cols else df, data.get("update_time", "")


def report_md():
    for n in ("recommendations_cn.md", "daily_report.md"):
        p = OUT / n
        if p.exists():
            return p.read_text(encoding="utf-8")
    return "_暂无日报_"


def backtest_info():
    for n in ("backtest_metrics.txt",):
        p = OUT / n
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                return "```json\n" + json.dumps(d, ensure_ascii=False, indent=2) + "\n```"
            except Exception:  # noqa: BLE001
                return p.read_text(encoding="utf-8")
    return "_暂无回测_"


def jobs_info():
    if not scheduler.running:
        return "调度器未运行"
    lines = []
    for j in scheduler.get_jobs():
        lines.append(f"- `{j.id}` → 下次: {j.next_run_time}")
    return "\n".join(lines) or "无任务"


def log_tail():
    p = OUT / "scheduler.log"
    return p.read_text(encoding="utf-8")[-4000:] if p.exists() else "_无日志_"


def manual_run(cmd):
    threading.Thread(target=job_train if cmd == "run" else job_refresh, daemon=True).start()
    return f"已后台触发 `{cmd}`，稍后刷新查看（日志见「状态」页）。"


# ---------- Gradio 看板 ----------
with gr.Blocks(title="A股+港股 AI 量化看板") as demo:
    gr.Markdown("# A股 + 港股 AI 量化看板\n复用当日模型，三概率（上涨/跑赢大盘/跑赢行业）+ 理由/风险。非投资建议。")
    with gr.Tab("推荐"):
        df_out = gr.Dataframe(headers=["code"], interactive=False)
        ts = gr.Markdown()
        b1 = gr.Button("🔄 刷新表格")
        b1.click(recs_df, outputs=[df_out, ts])
    with gr.Tab("日报"):
        gr.Markdown(report_md)
    with gr.Tab("回测"):
        gr.Markdown(backtest_info)
    with gr.Tab("状态"):
        gr.Markdown("### 调度\n- `train`：工作日 17:00 全流程训练\n- `refresh`：每 2h 复用模型刷新日报")
        gr.Markdown(jobs_info)
        with gr.Row():
            gr.Button("立即训练").click(lambda: manual_run("run"), outputs=gr.Markdown())
            gr.Button("立即刷新日报").click(lambda: manual_run("refresh"), outputs=gr.Markdown())
        gr.Textbox(label="最近日志(尾)", value=log_tail, lines=18)

    demo.load(recs_df, outputs=[df_out, ts])

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
