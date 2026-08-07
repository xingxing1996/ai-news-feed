"""仓库层：SQLite upsert + Parquet 读写 + DuckDB SQL。

- 元数据（stock / financial / news）走 SQLite（见 models.py）。
- 行情与分析层走 Parquet（列式、零基建、可被 DuckDB 当表查）。
- 未来换 Postgres/ClickHouse 时，只需替换本文件的实现，上层不动。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sqlalchemy import text

from ..config import get_settings
from .models import Base, get_engine, get_session, init_db

try:
    import duckdb  # 延迟导入失败也不影响 SQLite/Parquet
    _HAS_DUCKDB = True
except Exception:  # pragma: no cover
    _HAS_DUCKDB = False


# ---------------------------- SQLite upsert ---------------------------- #

def _columns(model) -> list[str]:
    return [c.name for c in model.__table__.columns if c.name != "id"]


def _conflict_keys(model, df_cols) -> list[str]:
    """从模型的约束里推断 upsert 冲突列（优先多列唯一键，否则单列）。"""
    candidates = []
    for cons in model.__table__.constraints:
        cnames = [c.name for c in cons.columns]
        if not cnames or cnames == ["id"]:
            continue
        if all(n in df_cols for n in cnames):
            candidates.append(cnames)
    # 优先列数最多的（更具体的业务键）
    candidates.sort(key=len, reverse=True)
    return candidates[0] if candidates else []


def upsert_dataframe(df: pd.DataFrame, model) -> int:
    """把 DataFrame 写入对应表（按业务唯一键 upsert）。

    要求 df 的列名与模型字段（除 id）一致。冲突键取自模型的 UniqueConstraint。
    """
    if df.empty:
        return 0
    init_db()
    cols = _columns(model)
    df = df[[c for c in cols if c in df.columns]].copy()

    table = model.__tablename__
    engine = get_engine()
    conflict = _conflict_keys(model, list(df.columns))

    placeholders = ",".join([f":{c}" for c in df.columns])
    update_clause = ",".join([f"{c}=excluded.{c}" for c in df.columns])
    sql = (
        f"INSERT INTO {table} ({','.join(df.columns)}) VALUES ({placeholders})"
        + (f" ON CONFLICT({','.join(conflict)}) DO UPDATE SET {update_clause}" if conflict else "")
    )
    records = df.where(pd.notna(df), None).to_dict("records")
    # SQLite 不能直接绑定 pd.Timestamp / numpy 标量，统一转 Python 原生类型
    def _san(v):
        if isinstance(v, pd.Timestamp):
            return v.to_pydatetime()
        if hasattr(v, "item"):  # numpy 标量
            try:
                return v.item()
            except Exception:  # noqa: BLE001
                return v
        return v

    records = [{k: _san(v) for k, v in r.items()} for r in records]
    with engine.begin() as conn:
        conn.execute(text(sql), records)
    return len(records)


# ---------------------------- Parquet 读写 ---------------------------- #

def warehouse_path(name: str) -> Path:
    """warehouse 下某张 parquet 的完整路径。"""
    return Path(get_settings().paths.warehouse_dir) / f"{name}.parquet"


def write_parquet(df: pd.DataFrame, name: str, partition: str | None = None) -> Path:
    """落盘到 warehouse。同名覆盖（按日期分区写则追加由调用方处理）。"""
    path = warehouse_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_parquet(name: str) -> pd.DataFrame:
    path = warehouse_path(name)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df


# ---------------------------- DuckDB 查询 ---------------------------- #

def duckdb_query(sql: str) -> pd.DataFrame:
    """对 warehouse 下的 parquet 跑 SQL。

    用法：duckdb_query("SELECT * FROM read_parquet('<warehouse>/daily.parquet') ...")
    或用 register_warehouse() 注册成表名后直接 FROM daily。
    """
    if not _HAS_DUCKDB:  # pragma: no cover
        raise RuntimeError("duckdb 未安装，无法运行 SQL；请先 pip install duckdb")
    con = duckdb.connect()
    _register(con)
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def _register(con) -> None:
    """把 warehouse 下每个 parquet 注册为同名表（无后缀）。"""
    wh = Path(get_settings().paths.warehouse_dir)
    for p in wh.glob("*.parquet"):
        con.register(p.stem, p.as_posix())


def export_table_to_parquet(model, name: str) -> Path:
    """把某张 SQLite 表导出为 parquet，便于 DuckDB 跨库分析。"""
    engine = get_engine()
    df = pd.read_sql_table(model.__tablename__, engine)
    return write_parquet(df, name)
