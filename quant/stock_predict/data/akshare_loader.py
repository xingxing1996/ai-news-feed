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
import os
import random
import requests as _req

# 强行清理残留死代理环境变量 (127.0.0.1)，保证网络请求直连不被卡死
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)

_orig_request = _req.Session.request

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15",
]


def _patched_request(self, method, url, **kwargs):
    # 自动清除无效死代理设置 (127.0.0.1)，确保网络直连
    proxies = kwargs.get("proxies") or {}
    if any("127.0.0.1" in str(v) for v in proxies.values()):
        kwargs["proxies"] = {}
    headers = kwargs.setdefault("headers", {})
    headers["User-Agent"] = random.choice(_USER_AGENTS)
    headers["Accept"] = "*/*"
    headers["Accept-Language"] = "zh-CN,zh;q=0.9,en;q=0.8"
    if "eastmoney" in str(url):
        headers["Referer"] = "https://quote.eastmoney.com/"
    # 强制设置硬超时 (connect 4s, read 10s)，杜绝死等挂起
    kwargs.setdefault("timeout", (4, 10))
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


def _call_ak(func, retries: int = 2, **kwargs):
    """调用 AKShare，带有防挂起硬超时与快速退避重试。"""
    for i in range(retries + 1):
        try:
            return func(**kwargs)
        except Exception as e:  # noqa: BLE001
            if i < retries:
                time.sleep(0.2 * (i + 1))
                continue
            log.debug("[akshare] 接口 %s 最终重试失败: %s", func.__name__, e)
            return None
    return None


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
    """下载前复权日线（A股 stock_zh_a_hist / 港股 stock_hk_daily + yfinance保底）。返回统一 schema。"""
    try:
        ak = _ak()
        if market == "hk":
            sym = code.split(".")[0].rjust(5, "0")  # 0700.HK → 00700
            df = None
            try:
                df = _call_ak(ak.stock_hk_daily, symbol=sym, adjust="qfq")
            except Exception:  # noqa: BLE001
                pass

            if df is None or df.empty:
                # 备用保底接口 1：stock_hk_hist
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

            # 备用保底接口 2：yfinance (100% 极速稳定保底)
            if df is None or df.empty:
                try:
                    from . import yfinance_loader as YL
                    return YL.fetch_daily(code, start, end)
                except Exception:  # noqa: BLE001
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
    """历史 PE/PB。A股 stock_zh_valuation_baidu / 港股 stock_hk_valuation_baidu + yfinance保底。"""
    try:
        ak = _ak()
        if market == "cn":
            sym = _to_ak_symbol(code)
            pe = _baidu_val(ak, "stock_zh_valuation_baidu", sym, "市盈率(TTM)")
            pb = _baidu_val(ak, "stock_zh_valuation_baidu", sym, "市净率")
        elif market == "hk":
            sym = code.split(".")[0].rjust(5, "0")  # 0700.HK → 00700
            pe, pb = None, None
            try:
                pe = _baidu_val(ak, "stock_hk_valuation_baidu", sym, "市盈率(TTM)")
                pb = _baidu_val(ak, "stock_hk_valuation_baidu", sym, "市净率")
            except Exception:  # noqa: BLE001
                pass
            if pe is None and pb is None:
                # yfinance 估值保底
                try:
                    from . import yfinance_loader as YL
                    return YL.fetch_valuation(code, start, end)
                except Exception:  # noqa: BLE001
                    return _empty_valuation()
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
            "profit_growth": _pick(df, ["净利润同比增长率", "净利润同比增长", "归母净利润同比增长率", "净利润增长率"]),
            "revenue_growth": _pick(df, ["营业收入同比增长率", "营业收入同比增长", "营业总收入同比增长率", "营收增长率"]),
        }
        # 归一化百分比 (若源数据是 45.2% 格式转为 0.452)
        if pd.notna(rec["profit_growth"]) and abs(rec["profit_growth"]) > 5.0:
            rec["profit_growth"] = rec["profit_growth"] / 100.0
        if pd.notna(rec["revenue_growth"]) and abs(rec["revenue_growth"]) > 5.0:
            rec["revenue_growth"] = rec["revenue_growth"] / 100.0

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
