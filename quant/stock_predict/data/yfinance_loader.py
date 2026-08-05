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
        try:
            tk = yf.Ticker(code)
            info = tk.info or {}
            shares = info.get("sharesOutstanding") or info.get("impliedShares") or info.get("floatShares")
            if shares and "close" in out.columns:
                out["market_cap"] = out["close"] * float(shares)
            else:
                mcap = info.get("marketCap")
                out["market_cap"] = float(mcap) if mcap else pd.NA
        except Exception:  # noqa: BLE001
            out["market_cap"] = pd.NA
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("[yfinance] %s 行情下载失败: %s", code, exc)
        return _empty_daily()


def fetch_valuation(code: str, start: str, end: str, market: str = "us") -> pd.DataFrame:
    """美股历史 PE/PB：用 yfinance 季度财报(净利润/净资产)+股价+股本自算。

    PE = close / EPS_TTM；PB = close / 每股净资产。
    点在时间近似：财报滞后 45 天才可知（shift +45d 后 forward-fill）。
    任何一步失败→返回空（美股估值因子缺省，模型照跑）。
    """
    import numpy as np

    try:
        yf = _yf()
        tk = yf.Ticker(code)
        px = yf.download(code, start=start, end=end, progress=False, auto_adjust=True)
        if px is None or px.empty or "close" not in px:
            return _empty_valuation()
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = [c[0] if isinstance(c, tuple) else c for c in px.columns]
        close = pd.to_numeric(px["close"], errors="coerce")
        close.index = pd.to_datetime(close.index)

        info = tk.info or {}
        shares = info.get("sharesOutstanding") or info.get("floatShares") or info.get("impliedShares")
        inc = tk.quarterly_income_stmt
        bal = tk.quarterly_balance_sheet
        if not shares or inc is None or inc.empty:
            return _empty_valuation()

        # TTM 净利润 = 最近4季滚动和 → EPS_TTM
        ni_row = next((inc.loc[n] for n in
                       ("Net Income", "Net Income Common Stockholders", "Net Income From Continuing Ops")
                       if n in inc.index), None)
        if ni_row is None:
            return _empty_valuation()
        ni = ni_row.dropna().sort_index()
        ttm = ni.rolling(4, min_periods=4).sum()
        eps = ttm / float(shares)                 # period_end -> EPS_TTM
        eps.index = eps.index + pd.Timedelta(days=45)   # 发布滞后
        eps_daily = eps.reindex(close.index, method="ffill")

        # 每股净资产 → PB
        pb_ps = None
        if bal is not None and not bal.empty:
            for n in ("Stockholders Equity", "Total Equity Gross", "Common Stock Equity"):
                if n in bal.index:
                    eq = bal.loc[n].dropna().sort_index() / float(shares)
                    eq.index = eq.index + pd.Timedelta(days=45)
                    pb_ps = eq.reindex(close.index, method="ffill")
                    break

        out = pd.DataFrame({"close": close})
        out["pe"] = out["close"] / eps_daily
        out["pb"] = out["close"] / pb_ps if pb_ps is not None else np.nan
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.reset_index()
        out = out.rename(columns={out.columns[0]: "date"})
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out["code"] = code
        out = out[(out["date"] >= start) & (out["date"] <= end)]
        return out[["date", "code", "pe", "pb"]]
    except Exception as exc:  # noqa: BLE001
        log.warning("[yfinance] %s PE/PB 自算失败，尝试 info 静态保底: %s", code, exc)
        try:
            yf = _yf()
            info = yf.Ticker(code).info or {}
            pe_val = _num(info, ["trailingPE", "forwardPE"])
            pb_val = _num(info, ["priceToBook"])
            if pe_val is not None or pb_val is not None:
                px = yf.download(code, start=start, end=end, progress=False)
                if px is not None and not px.empty:
                    dates = pd.to_datetime(px.index).strftime("%Y-%m-%d")
                    out = pd.DataFrame({"date": dates, "code": code})
                    out["pe"] = pe_val if pe_val is not None else np.nan
                    out["pb"] = pb_val if pb_val is not None else np.nan
                    return out[["date", "code", "pe", "pb"]]
        except Exception:  # noqa: BLE001
            pass
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
