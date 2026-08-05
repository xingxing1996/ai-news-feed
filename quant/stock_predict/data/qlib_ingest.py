"""把行情 Parquet 转成 Qlib ``.bin`` 数据（provider_uri）。

Qlib 的标准做法是用仓库里的 ``scripts/dump_bin.py`` 把 CSV 目录转成 .bin。
本模块：
1. 从 ``daily_price`` Parquet 生成「每只股票一个 CSV」的目录；
2. 自动定位 dump_bin.py（pip 装的 pyqlib 里路径不固定，故多处搜索）；
3. 以子进程执行 dump_all。

注意：Alpha158 因子在 ``features/alpha.py`` 里有 pandas 兜底实现，
即使此处 dump 失败，整条流水线仍可运行（只是用不了 Qlib 的因子引擎）。
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT, get_settings
from .warehouse import read_parquet

log = logging.getLogger(__name__)

# dump_bin 约定的字段（Qlib 表达式 $close 等以此命名）
QLIB_FIELDS = ["open", "high", "low", "close", "volume"]


def find_dump_bin() -> Path | None:
    """定位 dump_bin.py：优先项目自带（scripts/qlib/），其次 qlib 包内。"""
    candidates = [
        PROJECT_ROOT / "scripts" / "qlib" / "dump_bin.py",
        PROJECT_ROOT / "scripts" / "dump_bin.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    try:
        import qlib  # noqa: WPS433

        base = Path(qlib.__file__).resolve().parent
        candidates = [
            base / "scripts" / "dump_bin.py",
            base.parent / "scripts" / "dump_bin.py",
            *list(base.parent.glob("**/dump_bin.py")),
        ]
        for c in candidates:
            if c.exists():
                return c
    except Exception:  # noqa: BLE001
        pass
    return None


def prepare_csv_dir(out_dir: Path) -> Path:
    """从 daily_price.parquet 生成 per-symbol CSV 目录。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    # 清空旧的 csv
    for f in out_dir.glob("*.csv"):
        f.unlink()

    df = read_parquet("daily_price")
    if df.empty:
        raise RuntimeError("daily_price 为空，请先执行 ingest 采集数据。")

    df = df[["date", "code", *QLIB_FIELDS]].copy()
    df = df.dropna(subset=["close"])
    # Qlib 代码：去掉后缀点号可能更好，但保留原 code 也可；这里用去掉点号的紧凑名
    df["symbol"] = df["code"].str.replace(".", "_", regex=False)
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date")
        g[["date", *QLIB_FIELDS]].to_csv(out_dir / f"{sym}.csv", index=False)
    return out_dir


def qlib_symbol(code: str) -> str:
    return code.replace(".", "_")


def ingest_to_qlib() -> dict:
    """执行 dump_bin，把数据写入 settings.paths.qlib_dir。"""
    cfg = get_settings()
    qlib_dir = Path(cfg.paths.qlib_dir)
    qlib_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = Path(cfg.paths.raw_dir) / "csv_for_qlib"
    prepare_csv_dir(csv_dir)

    dump_bin = find_dump_bin()
    if dump_bin is None:
        msg = (
            "未找到 Qlib 的 dump_bin.py（pip 安装的 pyqlib 可能未附带 scripts）。\n"
            "可选方案：1) 从 https://github.com/microsoft/qlib 取 scripts/dump_bin.py 放到项目；\n"
            "          2) 直接使用本项目的 pandas 因子兜底（features/alpha.py），跳过 Qlib 数据层。\n"
            f"已生成 CSV 目录：{csv_dir}（可手动执行 dump_bin）。"
        )
        log.warning(msg)
        return {"ok": False, "reason": "dump_bin_not_found", "csv_dir": str(csv_dir), "qlib_dir": str(qlib_dir)}

    cmd = [
        sys.executable, str(dump_bin), "dump_all",
        "--data_path", str(csv_dir),
        "--qlib_dir", str(qlib_dir),
        "--include_fields", ",".join(QLIB_FIELDS),
        "--date_field_name", "date",
    ]
    log.info("执行 dump_bin: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return {"ok": False, "reason": str(exc)}
    if proc.returncode != 0:
        log.warning("dump_bin 失败 (rc=%s):\n%s", proc.returncode, proc.stderr[-2000:])
        return {"ok": False, "reason": "dump_bin_failed", "stderr": proc.stderr[-2000:],
                "csv_dir": str(csv_dir), "qlib_dir": str(qlib_dir)}

    return {"ok": True, "qlib_dir": str(qlib_dir), "csv_dir": str(csv_dir)}


def write_instruments_file(universe_df: pd.DataFrame) -> Path:
    """写 Qlib instruments 列表（dump_all 通常已生成，这里兜底）。"""
    cfg = get_settings()
    inst_dir = Path(cfg.paths.qlib_dir) / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    path = inst_dir / "all.txt"
    lines = [f"{qlib_symbol(r.code)}\t2018-01-01\t2099-12-31" for r in universe_df.itertuples(index=False)]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
