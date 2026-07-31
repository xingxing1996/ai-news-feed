"""统一日志：控制台 + logs/run.log（轮转）+ logs/errors.log（WARNING+）+ logs/status.jsonl。

所有脚本（main/picks/resolve_models/quant）启动时调 setup_logger()，即可：
  - 全量日志写 logs/run.log；
  - 报错写 logs/errors.log（打开这个就知道哪里出问题）；
  - 关键阶段写 logs/status.jsonl（一行一事件，便于扫描）；
  - 未捕获异常不会静默崩，会进 errors.log。
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

_BJ = timezone(timedelta(hours=8))
_ROOT = Path(__file__).resolve().parent.parent


def setup_logger(level: int = logging.INFO, logs_dir: str | Path | None = None) -> logging.Logger:
    """配置 root logger（库日志也会被捕获）。幂等。"""
    root = logging.getLogger()
    if getattr(setup_logger, "_done", False):
        return root
    setup_logger._done = True

    logs_dir = Path(logs_dir) if logs_dir else _ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    root.setLevel(level)
    # 控制台
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    # 全量
    rh = RotatingFileHandler(logs_dir / "run.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    rh.setFormatter(fmt)
    rh.setLevel(level)
    root.addHandler(rh)
    # 仅错误（方便感知）
    eh = RotatingFileHandler(logs_dir / "errors.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    eh.setFormatter(fmt)
    eh.setLevel(logging.WARNING)
    root.addHandler(eh)

    # 兜底：未捕获异常
    def _excepthook(exc_type, exc, tb):
        logging.getLogger("excepthook").error("未捕获异常", exc_info=(exc_type, exc, tb))

    sys.excepthook = _excepthook
    return root


def log_status(stage: str, ok: bool, msg: str = "", **stats) -> None:
    """追加一条结构化状态到 logs/status.jsonl。"""
    path = _ROOT / "logs" / "status.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.now(_BJ).strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "ok": bool(ok),
        "msg": msg,
        **stats,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
