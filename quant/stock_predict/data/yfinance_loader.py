"""港股 / 美股数据加载（yfinance）。

国内访问可能不稳定；失败返回空 DataFrame，由上层（含 synthetic 兜底）接管。
统一输出 schema 与 akshare_loader 一致。
"""
from __future__ import annotations

import os
import logging
import time

# 强行清理残留死代理环境变量 (127.0.0.1)，保证网络请求直连不被卡死
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)

import pandas as pd

log = logging.getLogger(__name__)

_YF = None


def _yf():
    global _YF
    if _YF is None:
        import yfinance as yf  # noqa: WPS433
        _YF = yf
    return _YF


def fetch_daily(code: str, start: str, end: str, market: str = "us") -> pd.DataFrame:
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
        # yfinance 的 info 只给当前股本/市值快照。把它回填到历史日线会产生
        # 明确的未来函数，因此在取得 PIT 历史股本前保持缺失。
        out["market_cap"] = pd.NA
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("[yfinance] %s 行情下载失败: %s", code, exc)
        return _empty_daily()


def _melt_yf_batch(df: pd.DataFrame) -> pd.DataFrame:
    """把 yf.download(多 code) 返回的 MultiIndex 宽表 ((field, ticker)) 融成长表(date, code, ohlcv)。"""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    if not isinstance(df.columns, pd.MultiIndex):
        return pd.DataFrame()  # 单 code 或非预期结构，交给逐只路径
    # 列形如 ("Date","") / ("Close","AAPL") → 拍成字符串列名
    df.columns = ["__date__" if (len(c) > 1 and c[1] == "") else f"{c[0]}__{c[1]}" for c in df.columns]
    codes = sorted({c.split("__", 1)[1] for c in df.columns if "__" in c})
    field_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    frames = []
    for code in codes:
        sub = {"date": df["__date__"]}
        ok = all(f"{f}__{code}" in df.columns for f in field_map)
        if not ok:
            continue
        for f, outf in field_map.items():
            sub[outf] = df[f"{f}__{code}"]
        sub["code"] = code
        frames.append(pd.DataFrame(sub))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).dropna(subset=["close"])
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["market_cap"] = pd.NA  # 批量无法取 per-Ticker 市值，宽基以价量为主干可接受
    return out[["date", "code", "open", "high", "low", "close", "volume", "market_cap"]]


def fetch_daily_batch(codes: list[str], start: str, end: str,
                      batch_size: int = 100, retries: int = 2) -> pd.DataFrame:
    """批量下载多只票日线：每 batch_size 只一次 yf.download，绕开逐只 YFRateLimitError。

    返回与 fetch_daily 同 schema（无 market_cap）。失败批次重试 retries 次、批间礼让。
    宽基(几百只)首选此路径，把 yfinance 请求从 N 次降到 N/batch_size 次。
    """
    yf = _yf()
    codes = [c for c in codes if c]
    parts = []
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        for attempt in range(retries + 1):
            try:
                df = yf.download(batch, start=start, end=end, progress=False,
                                 auto_adjust=True, group_by="column", threads=True)
                melted = _melt_yf_batch(df)
                if not melted.empty:
                    parts.append(melted)
                    log.info("[yfinance] 批量下载第%d批 %d 只 → %d 行",
                             i // batch_size + 1, len(batch), len(melted))
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("[yfinance] 批量下载第%d批失败(attempt %d/%d): %s",
                            i // batch_size + 1, attempt + 1, retries + 1, exc)
                if attempt < retries:
                    time.sleep(3)
        time.sleep(0.5)  # 批间礼让
    if not parts:
        return _empty_daily()
    return pd.concat(parts, ignore_index=True)


def fetch_valuation(code: str, start: str, end: str, market: str = "us") -> pd.DataFrame:
    """美股历史 PE/PB：用 yfinance 季度财报(净利润/净资产)+股价+股本自算。

    PE = close / EPS_TTM；PB = close / 每股净资产。
    点在时间近似：财报滞后 45 天才可知（shift +45d 后 forward-fill）。
    任何一步失败→返回空（美股估值因子缺省，模型照跑）。
    """
    # yfinance 财报和 sharesOutstanding 都是当前快照，不能可靠还原历史 EPS/BPS。
    # 宁可缺失，也不能把今天的估值或股本注入历史训练样本。
    log.info("[yfinance] %s 跳过非 PIT 历史估值", code)
    return _empty_valuation()


def _fetch_valuation_unsafe_legacy(code: str, start: str, end: str) -> pd.DataFrame:
    """保留旧解析实现供迁移参考；不得在训练或生产路径调用。"""
    import numpy as np

    try:
        yf = _yf()
        tk = yf.Ticker(code)
        px = yf.download(code, start=start, end=end, progress=False, auto_adjust=True)
        if px is None or px.empty:
            return _empty_valuation()
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = [c[0] if isinstance(c, tuple) else c for c in px.columns]
        px.columns = [str(c).lower() for c in px.columns]
        if "close" not in px:
            return _empty_valuation()
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
        # TTM 净利润 = 最近4季滚动和（若不足4季按已存在季数做年化） → EPS_TTM
        ni_row = next((inc.loc[n] for n in
                       ("Net Income", "Net Income Common Stockholders", "Net Income From Continuing Ops")
                       if n in inc.index), None)
        if ni_row is None:
            return _empty_valuation()
        ni = pd.to_numeric(ni_row, errors="coerce").dropna().sort_index()
        if ni.empty:
            return _empty_valuation()
        cnt = ni.rolling(4, min_periods=1).count()
        sums = ni.rolling(4, min_periods=1).sum()
        ttm = sums * (4.0 / cnt.clip(lower=1))
        eps = ttm / float(shares)                 # period_end -> EPS_TTM
        eps.index = eps.index + pd.Timedelta(days=45)   # 发布滞后
        eps_daily = eps.reindex(close.index, method="ffill")

        # 每股净资产 → PB
        pb_ps = None
        if bal is not None and not bal.empty:
            for n in ("Stockholders Equity", "Total Equity Gross", "Common Stock Equity"):
                if n in bal.index:
                    eq = pd.to_numeric(bal.loc[n], errors="coerce").dropna().sort_index() / float(shares)
                    if not eq.empty:
                        eq.index = eq.index + pd.Timedelta(days=45)
                        pb_ps = eq.reindex(close.index, method="ffill")
                        break

        out = pd.DataFrame({"close": close})
        out["pe"] = out["close"] / eps_daily
        out["pb"] = out["close"] / pb_ps if pb_ps is not None else np.nan
        out = out.replace([np.inf, -np.inf], np.nan)

        # 静态 info 保底：如果算出来的 pe/pb 全空，用 info 补充
        if out["pe"].isna().all() and _num(info, ["trailingPE", "forwardPE"]) is not None:
            out["pe"] = float(_num(info, ["trailingPE", "forwardPE"]))
        if out["pb"].isna().all() and _num(info, ["priceToBook"]) is not None:
            out["pb"] = float(_num(info, ["priceToBook"]))

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


def _yf_stmt_row(stmt, names: list[str]):
    """从 yfinance 季度报表(行=指标,列=报告期 Timestamp)按候选名取首行(Series)；无则 None。"""
    if stmt is None or getattr(stmt, "empty", True):
        return None
    idx = stmt.index.astype(str)
    for n in names:
        hits = stmt.loc[idx.str.contains(n, na=False, regex=False)]
        if not hits.empty:
            return hits.iloc[0]
    return None


def _yf_stmt(tk, primary: str, fallback: str):
    """优先用新版 *_stmt 接口，回退到旧版 *_financials/*_balance_sheet/*_cashflow。"""
    s = getattr(tk, primary, None)
    if s is None or getattr(s, "empty", True):
        s = getattr(tk, fallback, None)
    return s


def _yf_quarterly_rows(tk, code: str) -> list[dict]:
    """从季度三大表拼多期 PIT 财务行。pub_date = 报告期 + (年报90天 / 季报45天)。
    严格 Point-in-Time：每期只用该期已披露数据，绝不把今天的快照盖到历史。"""
    qf = _yf_stmt(tk, "quarterly_income_stmt", "quarterly_financials")
    qb = _yf_stmt(tk, "quarterly_balance_sheet", "quarterly_balance_sheet")
    qc = _yf_stmt(tk, "quarterly_cashflow", "quarterly_cashflow")
    rev = _yf_stmt_row(qf, ["Total Revenue", "Revenue"])
    ni = _yf_stmt_row(qf, ["Net Income", "NetIncome"])
    gp = _yf_stmt_row(qf, ["Gross Profit", "GrossProfit"])
    eq = _yf_stmt_row(qb, ["Stockholders Equity", "Total Equity", "Common Stock Equity", "Total Stockholder Equity"])
    ocf = _yf_stmt_row(qc, ["Operating Cash Flow", "Total Cash From Operating Activities", "OperatingCashflow"])
    periods = set()
    for s in (rev, ni, gp, eq, ocf):
        if s is not None:
            try:
                periods |= set(s.index)
            except Exception:  # noqa: BLE001
                pass
    if not periods:
        return []
    periods = sorted(periods, key=lambda p: pd.Timestamp(p))  # 升序，便于按位置取去年同期

    def _g(s, p):
        if s is None:
            return pd.NA
        try:
            return pd.to_numeric(s.get(p), errors="coerce")
        except Exception:  # noqa: BLE001
            return pd.NA

    rows = []
    for i, p in enumerate(periods):
        rp = pd.Timestamp(p)
        if pd.isna(rp):
            continue
        pub = rp + pd.Timedelta(days=90 if rp.month == 12 else 45)
        revenue = _g(rev, p)
        profit = _g(ni, p)
        gpv = _g(gp, p)
        ocfv = _g(ocf, p)
        gross_margin = gpv / revenue if (pd.notna(gpv) and pd.notna(revenue) and revenue) else pd.NA
        # YoY：去年同期(约 4 个季度前)，没有则留 NA（quality 兜底再算）
        profit_growth = revenue_growth = pd.NA
        if i >= 4:
            yoy = periods[i - 4]
            r_prev, p_prev = _g(rev, yoy), _g(ni, yoy)
            if pd.notna(revenue) and pd.notna(r_prev) and r_prev:
                revenue_growth = revenue / r_prev - 1
            if pd.notna(profit) and pd.notna(p_prev) and p_prev:
                profit_growth = profit / p_prev - 1
        rows.append({
            "code": code,
            "report_period": rp.strftime("%Y-%m-%d"),
            "pub_date": pub.strftime("%Y-%m-%d"),
            "revenue": revenue,
            "profit": profit,
            "roe": pd.NA,  # 季度 ROE 年化不可靠，宁缺毋滥（避免错误值污染截面 rank）
            "gross_margin": gross_margin,
            "cashflow": ocfv,
            "pe": pd.NA,
            "pb": pd.NA,
            "profit_growth": profit_growth,
            "revenue_growth": revenue_growth,
        })
    return rows


def fetch_financial(code: str, market: str = "us") -> pd.DataFrame:
    try:
        yf = _yf()
        tk = yf.Ticker(code)
        rows = _yf_quarterly_rows(tk, code)
        if rows:
            return pd.DataFrame(rows)
        return _empty_fin()  # 季度表取不到则返回空（不回退 today，避免历史假死/泄漏）
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
