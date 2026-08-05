"""SQLAlchemy 表结构（对应设计文档第 4 节）。

表：
  stock         code / name / market / industry / country
  daily_price   date / code / ohlcv / market_cap
  financial     code / report_period / 发布日 / revenue / profit / roe / gross_margin / cashflow / pe / pb
  news          id / stock / title / content / publish_time / source / sentiment / impact_score

说明：所有表用统一的主键约束，便于 upsert。code 为跨市场统一代码
（如 600519.SH / 0700.HK / AAPL）。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from ..config import get_settings


class Base(DeclarativeBase):
    pass


class Stock(Base):
    __tablename__ = "stock"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[Optional[str]] = mapped_column(String(64))
    market: Mapped[str] = mapped_column(String(8), index=True)  # cn / hk / us
    industry: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    country: Mapped[Optional[str]] = mapped_column(String(32))

    __table_args__ = (UniqueConstraint("code", name="uq_stock_code"),)


class DailyPrice(Base):
    __tablename__ = "daily_price"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String(10), index=True)  # YYYY-MM-DD，便于分组
    code: Mapped[str] = mapped_column(String(32), index=True)
    open: Mapped[Optional[float]] = mapped_column(Float)
    high: Mapped[Optional[float]] = mapped_column(Float)
    low: Mapped[Optional[float]] = mapped_column(Float)
    close: Mapped[Optional[float]] = mapped_column(Float)
    volume: Mapped[Optional[float]] = mapped_column(Float)
    market_cap: Mapped[Optional[float]] = mapped_column(Float)

    __table_args__ = (UniqueConstraint("date", "code", name="uq_daily_date_code"),)


class Financial(Base):
    """财务数据。report_period 为报告期（如 2023-Q4），pub_date 为发布日（防未来函数）。"""

    __tablename__ = "financial"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), index=True)
    report_period: Mapped[Optional[str]] = mapped_column(String(10), index=True)
    pub_date: Mapped[Optional[str]] = mapped_column(String(10), index=True)
    revenue: Mapped[Optional[float]] = mapped_column(Float)
    profit: Mapped[Optional[float]] = mapped_column(Float)  # 归母净利润
    roe: Mapped[Optional[float]] = mapped_column(Float)
    gross_margin: Mapped[Optional[float]] = mapped_column(Float)
    cashflow: Mapped[Optional[float]] = mapped_column(Float)  # 经营现金流
    pe: Mapped[Optional[float]] = mapped_column(Float)        # 估值快照（随行情）
    pb: Mapped[Optional[float]] = mapped_column(Float)

    __table_args__ = (UniqueConstraint("code", "report_period", name="uq_fin_code_period"),)


class News(Base):
    __tablename__ = "news"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    title: Mapped[Optional[str]] = mapped_column(String(512))
    content: Mapped[Optional[str]] = mapped_column(String)
    publish_time: Mapped[Optional[datetime]] = mapped_column(index=True)
    source: Mapped[Optional[str]] = mapped_column(String(64))
    sentiment: Mapped[Optional[str]] = mapped_column(String(16))  # positive/negative/neutral
    impact_score: Mapped[Optional[float]] = mapped_column(Float)  # 0~1，LLM 事件抽取产出


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        path = get_settings().paths.sqlite_path
        _engine = create_engine(f"sqlite:///{path}", future=True)
    return _engine


def get_session():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionLocal()


def init_db() -> None:
    """建表（幂等）。"""
    Base.metadata.create_all(get_engine())
