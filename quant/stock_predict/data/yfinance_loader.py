"""港股 / 美股数据加载（yfinance）。

国内访问可能不稳定；失败返回空 DataFrame，由上层（含 synthetic 兜底）接管。
统一输出 schema 与 akshare_loader 一致。
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

_YF = None


def _yf():
    global _YF
    if _YF is None:
        import yfinance as yf  # noqa: WPS433
        _YF = yf
    return _YF


def fetch_daily(code: str, start: str, end: str) -> pd.DataFrame:
    try:
        yf = _yf()
        df = yf.download(code, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return _empty_daily()
        df = df.reset_index()
        # yfinance 列可能是 MultiIndex（新版本），拍平
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        colmap = {"Date": "date", "Open": "open", "High": "high", "Low": "low",
                  "Close": "close", "Volume": "volume"}
        df = df.rename(columns=colmap)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["code"] = code
        out = df[["date", "code", "open", "high", "low", "close", "volume"]].copy()
        out["market_cap"] = pd.NA
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("[yfinance] %s 行情下载失败: %s", code, exc)
        return _empty_daily()


def fetch_valuation(code: str, start: str, end: str) -> pd.DataFrame:
    """yfinance 不直接提供历史 PE/PB 序列；这里返回空，由上层用财务快照近似或跳过估值因子。"""
    return _empty_valuation()


def fetch_financial(code: str, market: str = "us") -> pd.DataFrame:
    try:
        yf = _yf()
        tk = yf.Ticker(code)
        info = tk.info or {}
        today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        rec = {
            "code": code,
            "report_period": None,
            "pub_date": today,
            "revenue": _num(info, ["totalRevenue"]),
            "profit": _num(info, ["netIncomeToCommon"]),
            "roe": _num(info, ["returnOnEquity"]),
            "gross_margin": _num(info, ["grossMargins"]),
            "cashflow": _num(info, ["operatingCashflow"]),
            "pe": _num(info, ["trailingPE"]),
            "pb": _num(info, ["priceToBook"]),
        }
        return pd.DataFrame([rec])
    except Exception as exc:  # noqa: BLE001
        log.warning("[yfinance] %s 财务下载失败: %s", code, exc)
        return _empty_fin()


def _num(d: dict, keys: list[str]):
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                return pd.NA
    return pd.NA


def _empty_daily() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "code", "open", "high", "low", "close", "volume", "market_cap"])


def _empty_valuation() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "code", "pe", "pb"])


def _empty_fin() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["code", "report_period", "pub_date", "revenue", "profit", "roe",
                 "gross_margin", "cashflow", "pe", "pb"]
    )
