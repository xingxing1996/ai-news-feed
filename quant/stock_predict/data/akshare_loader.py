"""A股数据加载（AKShare）。

AKShare 函数签名随版本变化频繁，这里做防御式封装：
- 失败不抛异常，返回空 DataFrame 并打日志（不让单只股票中断整条流水线）。
- 统一输出 schema：date / code / ohlcv / market_cap。
- 财务统一输出 schema：code / report_period / pub_date / revenue / profit / roe / gross_margin / cashflow / pe / pb。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

import pandas as pd

# ---- monkey-patch requests 加真实浏览器 UA 与 Header（降低东财/百度拒连/限流概率）----
import random
import requests as _req

_orig_request = _req.Session.request
_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"


def _patched_request(self, method, url, **kwargs):
    headers = kwargs.setdefault("headers", {})
    if not headers.get("User-Agent"):
        headers["User-Agent"] = _BROWSER_UA
    if not headers.get("Accept-Language"):
        headers["Accept-Language"] = "zh-CN,zh;q=0.9,en;q=0.8"
    return _orig_request(self, method, url, **kwargs)


_req.Session.request = _patched_request

log = logging.getLogger(__name__)

_AK = None  # 延迟导入


def _ak():
    global _AK
    if _AK is None:
        import akshare as ak  # noqa: WPS433
        _AK = ak
    return _AK


def _call_ak(func, retries: int = 4, **kwargs):
    """调用 AKShare，连接被拒/断开/超时自动重试（带随机指数退避，最高 4 次）。"""
    for i in range(retries + 1):
        try:
            return func(**kwargs)
        except Exception as e:  # noqa: BLE001
            s = str(e)
            is_net_err = any(kw in s or kw in s.lower() for kw in (
                "connection", "remotedisconnected", "timeout", "reset", "aborted", "closed", "refused"
            ))
            if i < retries and is_net_err:
                # 随机指数退避（Jitter）：1.5^i + [0.5, 1.5] 秒
                wait = (1.5 ** i) + random.uniform(0.5, 1.5)
                log.debug("[akshare] 接口 %s 发生连接中断 (%s)，休眠 %.1fs 后第 %d 次重试...", func.__name__, s, wait, i + 1)
                time.sleep(wait)
                continue
            raise
    return func(**kwargs)


def _to_ak_symbol(code: str) -> str:
    """600519.SH / 000001.SZ → 600519 / 000001。"""
    return code.split(".")[0]


_ETF_PREFIXES = {"510", "511", "512", "513", "515", "516", "518", "588", "159", "150"}


def _is_etf(code: str) -> bool:
    """识别 A股 ETF 代码（510xxx/512xxx/159xxx/588xxx 等）。"""
    s = code.split(".")[0]
    return len(s) == 6 and s[:3] in _ETF_PREFIXES


def _to_date(s) -> str:
    return pd.to_datetime(s).strftime("%Y-%m-%d")


def fetch_daily(code: str, start: str, end: str, market: str = "cn") -> pd.DataFrame:
    """下载前复权日线（A股 stock_zh_a_hist / 港股 stock_hk_daily）。返回统一 schema。"""
    try:
        ak = _ak()
        if market == "hk":
            sym = code.split(".")[0].rjust(5, "0")  # 0700.HK → 00700
            df = _call_ak(ak.stock_hk_daily, symbol=sym, adjust="qfq")
            if df is None or df.empty:
                # 备用保底接口：stock_hk_hist
                try:
                    df = _call_ak(
                        ak.stock_hk_hist,
                        symbol=sym,
                        period="daily",
                        start_date=pd.to_datetime(start).strftime("%Y%m%d"),
                        end_date=pd.to_datetime(end).strftime("%Y%m%d"),
                        adjust="qfq",
                    )
                    if df is not None and not df.empty:
                        colmap = {
                            "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
                            "收盘": "close", "成交量": "volume",
                        }
                        df = df.rename(columns=colmap)
                except Exception:  # noqa: BLE001
                    pass
            if df is None or df.empty:
                return _empty_daily()
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            out = df[["date", "open", "high", "low", "close", "volume"]].copy()
            out["code"] = code
            out["market_cap"] = pd.NA
            return out[["date", "code", "open", "high", "low", "close", "volume", "market_cap"]]

        # A股 / A股ETF（ETF 用 fund_etf_hist_em）
        sym = _to_ak_symbol(code)
        if _is_etf(code):
            df = _call_ak(
                ak.fund_etf_hist_em,
                symbol=sym,
                period="daily",
                start_date=pd.to_datetime(start).strftime("%Y%m%d"),
                end_date=pd.to_datetime(end).strftime("%Y%m%d"),
                adjust="qfq",
            )
        else:
            df = _call_ak(
                ak.stock_zh_a_hist,
                symbol=sym,
                period="daily",
                start_date=pd.to_datetime(start).strftime("%Y%m%d"),
                end_date=pd.to_datetime(end).strftime("%Y%m%d"),
                adjust="qfq",
            )
        if df is None or df.empty:
            return _empty_daily()
        colmap = {
            "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
            "收盘": "close", "成交量": "volume", "成交额": "amount",
        }
        df = df.rename(columns=colmap)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["code"] = code
        out = df[["date", "code", "open", "high", "low", "close", "volume"]].copy()
        out["market_cap"] = pd.NA
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("[akshare] %s 行情下载失败: %s", code, exc)
        return _empty_daily()


def fetch_valuation(code: str, start: str, end: str, market: str = "cn") -> pd.DataFrame:
    """历史 PE/PB。A股 stock_zh_valuation_baidu / 港股 stock_hk_valuation_baidu（同源百度）。"""
    try:
        ak = _ak()
        if market == "cn":
            sym = _to_ak_symbol(code)
            pe = _baidu_val(ak, "stock_zh_valuation_baidu", sym, "市盈率(TTM)")
            pb = _baidu_val(ak, "stock_zh_valuation_baidu", sym, "市净率")
        elif market == "hk":
            sym = code.split(".")[0].rjust(5, "0")  # 0700.HK → 00700
            pe = _baidu_val(ak, "stock_hk_valuation_baidu", sym, "市盈率(TTM)")
            pb = _baidu_val(ak, "stock_hk_valuation_baidu", sym, "市净率")
        else:
            return _empty_valuation()
        if pe is None and pb is None:
            return _empty_valuation()
        df = pd.merge(pe, pb, on="date", how="outer") if (pe is not None and pb is not None) else (pe or pb)
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df.assign(code=code)[["date", "code", "pe", "pb"]]
    except Exception as exc:  # noqa: BLE001
        log.debug("[akshare] %s 估值下载失败: %s", code, exc)
        return _empty_valuation()


def _baidu_val(ak, func_name: str, sym: str, indicator: str):
    """从百度估值接口（zh/hk）取单个指标，返回 [date, pe|pb]。"""
    try:
        func = getattr(ak, func_name)
        df = _call_ak(func, symbol=sym, indicator=indicator, period="全部")
        if df is None or df.empty:
            return None
        col = "pe" if "盈" in indicator else "pb"
        df = df.rename(columns={df.columns[0]: "date", df.columns[-1]: col})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df[["date", col]]
    except Exception:  # noqa: BLE001
        return None


def fetch_northbound(code: str, start: str, end: str, market: str = "cn") -> pd.DataFrame:
    """北向资金（沪深港通）个股持股数量历史。仅 A股（stock_hsgt_individual_em）。"""
    if market != "cn":
        return _empty_nb()
    try:
        ak = _ak()
        sym = _to_ak_symbol(code)
        df = _call_ak(ak.stock_hsgt_individual_em, symbol=sym)
        if df is None or df.empty or "持股日期" not in df.columns:
            return _empty_nb()
        df = df.rename(columns={"持股日期": "date", "持股数量": "north_shares"})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df[["date", "north_shares"]].assign(code=code)[["date", "code", "north_shares"]]
    except Exception as exc:  # noqa: BLE001
        log.debug("[akshare] %s 北向资金失败: %s", code, exc)
        return _empty_nb()


def _empty_nb() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "code", "north_shares"])


def fetch_financial(code: str, market: str = "cn") -> pd.DataFrame:
    """下载财务摘要。仅 A股 用 stock_financial_abstract；港股与 ETF 无财报，直接返回空。"""
    if market != "cn" or _is_etf(code):
        return _empty_fin()  # 港股与 ETF 无上市公司财报，直接返回空，避免无意义的接口报错警告
    try:
        ak = _ak()
        sym = _to_ak_symbol(code)
        df = _call_ak(ak.stock_financial_abstract, symbol=sym)
        if df is None or df.empty:
            return _empty_fin()
        return _normalize_financial_ak(df, code)
    except Exception as exc:  # noqa: BLE001
        log.warning("[akshare] %s 财务下载失败: %s", code, exc)
        return _empty_fin()


def _normalize_financial_ak(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """尽力把 AKShare 财务表拍平成统一 schema（容忍字段缺失）。"""
    # 不同版本列名差异大；这里用通用兜底：取不到就留空
    rows = []
    today = datetime.today().strftime("%Y-%m-%d")
    # 尝试取最近一期作为代表（避免被复杂结构拖垮）
    try:
        rec = {
            "code": code,
            "report_period": None,
            "pub_date": today,
            "revenue": _pick(df, ["营业总收入", "营业收入", "总营收"]),
            "profit": _pick(df, ["净利润", "归母净利润"]),
            "roe": _pick(df, ["净资产收益率", "ROE"]),
            "gross_margin": _pick(df, ["销售毛利率", "毛利率"]),
            "cashflow": _pick(df, ["经营活动现金流量净额", "经营现金流净额"]),
            "pe": pd.NA,
            "pb": pd.NA,
        }
        rows.append(rec)
    except Exception:  # noqa: BLE001
        return _empty_fin()
    return pd.DataFrame(rows)


def _pick(df: pd.DataFrame, names: list[str]):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n].iloc[0], errors="coerce")
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


def fetch_fund_flow(code: str, start: str, end: str, market: str = "cn") -> pd.DataFrame:
    """A股个股主力资金流向（stock_individual_fund_flow）。港股/美股退化返回空。"""
    if market != "cn" or _is_etf(code):
        return pd.DataFrame(columns=["date", "code", "main_fund_ratio", "super_fund_ratio"])
    try:
        ak = _ak()
        sym = _to_ak_symbol(code)
        mkt = "sh" if code.endswith(".SH") else "sz"
        df = _call_ak(ak.stock_individual_fund_flow, stock=sym, market=mkt)
        if df is None or df.empty or "日期" not in df.columns:
            return pd.DataFrame(columns=["date", "code", "main_fund_ratio", "super_fund_ratio"])
        df = df.rename(columns={
            "日期": "date",
            "主力净流入-净占比": "main_fund_ratio",
            "超大单净流入-净占比": "super_fund_ratio"
        })
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["main_fund_ratio"] = pd.to_numeric(df["main_fund_ratio"], errors="coerce") / 100.0
        df["super_fund_ratio"] = pd.to_numeric(df["super_fund_ratio"], errors="coerce") / 100.0
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df.assign(code=code)[["date", "code", "main_fund_ratio", "super_fund_ratio"]]
    except Exception as exc:  # noqa: BLE001
        log.debug("[akshare] %s 资金流向下载失败: %s", code, exc)
        return pd.DataFrame(columns=["date", "code", "main_fund_ratio", "super_fund_ratio"])


def fetch_cyq(code: str, start: str, end: str, market: str = "cn") -> pd.DataFrame:
    """A股筹码分布（stock_cyq_em：获利比例 / 90%集中度）。港股/美股退化返回空。"""
    if market != "cn" or _is_etf(code):
        return pd.DataFrame(columns=["date", "code", "chip_profit_ratio", "chip_concentration_90"])
    try:
        ak = _ak()
        sym = _to_ak_symbol(code)
        df = _call_ak(ak.stock_cyq_em, symbol=sym)
        if df is None or df.empty or "日期" not in df.columns:
            return pd.DataFrame(columns=["date", "code", "chip_profit_ratio", "chip_concentration_90"])
        df = df.rename(columns={
            "日期": "date",
            "获利比例": "chip_profit_ratio",
            "90集中度": "chip_concentration_90"
        })
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["chip_profit_ratio"] = pd.to_numeric(df["chip_profit_ratio"], errors="coerce")
        df["chip_concentration_90"] = pd.to_numeric(df["chip_concentration_90"], errors="coerce")
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df.assign(code=code)[["date", "code", "chip_profit_ratio", "chip_concentration_90"]]
    except Exception as exc:  # noqa: BLE001
        log.debug("[akshare] %s 筹码分布下载失败: %s", code, exc)
        return pd.DataFrame(columns=["date", "code", "chip_profit_ratio", "chip_concentration_90"])
