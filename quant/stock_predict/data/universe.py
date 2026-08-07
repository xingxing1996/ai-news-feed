"""股票池解析与入库。

读 ``config/universe.yaml``，按 demo_mode 截断，写入 ``stock`` 表，
并返回统一格式的 DataFrame。支持把 us 段动态替换为 S&P 500 成分股（宽基宇宙）。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from ..config import PROJECT_ROOT, get_settings
from .models import Stock
from .warehouse import upsert_dataframe

log = logging.getLogger(__name__)

_COUNTRY = {"cn": "中国", "hk": "中国香港", "us": "美国", "kr": "韩国"}

_SP500_CACHE: pd.DataFrame | None = None  # 进程内缓存
_CSI300_CACHE: pd.DataFrame | None = None


def _cn_exchange_suffix(code: str) -> str:
    """A 股裸代码(如 600118/000858/688981/830799) → 交易所后缀。"""
    c = str(code).strip().zfill(6)
    if c[:2] in ("60", "68") or c[:3] in ("118", "110", "113", "124", "567", "588"):
        return ".SH"
    if c[:2] in ("00", "30") or c[:3] in ("123", "127", "128", "159"):
        return ".SZ"
    if c[0] in ("8", "4") or c[:2] == "92":
        return ".BJ"
    return ".SH"


def load_universe_file() -> dict:
    path = PROJECT_ROOT / get_settings().universe.config_file
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def sp500_constituents(force_refresh: bool = False, max_age_days: int = 30) -> pd.DataFrame:
    """S&P 500 成分股：抓 Wikipedia 表 + 本地缓存。返回列 code/name/market/industry/country。

    - yfinance 用 "-" 表示分级/含点符号（如 BRK.B → BRK-B），故 code 做 "."→"-" 替换。
    - 抓取失败返回空 DataFrame，调用方回退静态 yaml（不抛错）。
    """
    global _SP500_CACHE
    if _SP500_CACHE is not None and not force_refresh:
        return _SP500_CACHE
    cache_path = PROJECT_ROOT / "data" / "cache" / "sp500_constituents.csv"
    # 本地缓存命中
    if not force_refresh and cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < max_age_days:
            try:
                df = pd.read_csv(cache_path)
                if not df.empty:
                    _SP500_CACHE = df
                    return df
            except Exception:  # noqa: BLE001
                pass
    # 抓 Wikipedia（须带正规 User-Agent，否则 GHA/默认 urllib UA 被 403）
    try:
        import requests
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        ua = {"User-Agent": "quant-stock-predict/1.0 (+https://github.com/xingxing1996/ai-news-feed; equity research)"}
        resp = requests.get(url, headers=ua, timeout=25)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        tbl = tables[0]
        sym_col = next(c for c in tbl.columns if str(c).lower().startswith("symbol"))
        name_col = next((c for c in tbl.columns if "security" in str(c).lower()), sym_col)
        sector_col = next((c for c in tbl.columns if "gics sector" in str(c).lower()), None)
        out = pd.DataFrame({
            "code": tbl[sym_col].astype(str).str.strip().str.replace(".", "-", regex=False),
            "name": tbl[name_col].astype(str).str.strip(),
            "market": "us",
            "industry": tbl[sector_col].astype(str).str.strip() if sector_col else "未知",
            "country": _COUNTRY.get("us"),
        })
        out = out[out["code"].str.len() > 0].drop_duplicates("code").reset_index(drop=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_path, index=False)
        _SP500_CACHE = out
        log.info("[universe] S&P 500 成分股 %d 只（Wikipedia 抓取）", len(out))
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("[universe] S&P 500 抓取失败，回退静态 yaml：%s", exc)
        return pd.DataFrame()


def csi300_constituents(force_refresh: bool = False, max_age_days: int = 30) -> pd.DataFrame:
    """沪深 300 成分股：akshare index_stock_cons + 本地缓存。返回列 code/name/market/industry/country。

    akshare 该接口不带行业，统一记 industry="沪深300"（label 退化为"跑赢沪深300截面"，对 ranker 有效；
    细行业后续可用 akshare 板块接口批量补）。code 加交易所后缀(600118→600118.SH)。失败返回空表→回退。
    """
    global _CSI300_CACHE
    if _CSI300_CACHE is not None and not force_refresh:
        return _CSI300_CACHE
    cache_path = PROJECT_ROOT / "data" / "cache" / "csi300_constituents.csv"
    if not force_refresh and cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < max_age_days:
            try:
                df = pd.read_csv(cache_path)
                if not df.empty:
                    _CSI300_CACHE = df
                    return df
            except Exception:  # noqa: BLE001
                pass
    try:
        import akshare as ak
        raw = ak.index_stock_cons(symbol="000300")
        code_col = next(c for c in raw.columns if "代码" in str(c))
        name_col = next((c for c in raw.columns if "名称" in str(c)), code_col)
        out = pd.DataFrame({
            "code": raw[code_col].astype(str).str.strip().str.zfill(6).map(lambda c: c + _cn_exchange_suffix(c)),
            "name": raw[name_col].astype(str).str.strip(),
            "market": "cn",
            "industry": "沪深300",
            "country": _COUNTRY.get("cn"),
        })
        out = out[out["code"].str.len() > 0].drop_duplicates("code").reset_index(drop=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_path, index=False)
        _CSI300_CACHE = out
        log.info("[universe] 沪深300 成分股 %d 只（akshare）", len(out))
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("[universe] 沪深300 抓取失败，回退静态 yaml：%s", exc)
        return pd.DataFrame()


def resolve_universe(demo: bool | None = None) -> pd.DataFrame:
    """返回统一格式的股票池 DataFrame。

    列：code, name, market, industry, country
    demo 为 None 时取 settings.universe.demo_mode。
    当 settings.universe.us_benchmark == "sp500" 时，us 段动态用 S&P 500 成分股（宽基）。
    """
    cfg = get_settings()
    demo = cfg.universe.demo_mode if demo is None else demo
    raw = load_universe_file()

    demo_size = dict(cfg.universe.get("demo_size", {})) or {}
    # 按市场过滤（拆分训练用：GHA 只跑 us/kr，中国机器跑 cn/hk）
    only_markets = cfg.universe.get("markets") or ["cn", "hk", "us", "kr"]
    us_benchmark = cfg.universe.get("us_benchmark")
    cn_benchmark = cfg.universe.get("cn_benchmark")
    rows: list[dict] = []
    for market in ("cn", "hk", "us", "kr"):
        if market not in only_markets:
            continue
        # cn 宽基：动态沪深300 成分股
        if market == "cn" and cn_benchmark == "csi300":
            cs = csi300_constituents()
            if not cs.empty:
                if demo and "cn" in demo_size:
                    cs = cs.head(int(demo_size["cn"]))
                for _, r in cs.iterrows():
                    rows.append({"code": r["code"], "name": r.get("name"), "market": "cn",
                                 "industry": r.get("industry"), "country": _COUNTRY.get("cn")})
                continue
            # 抓取失败 → 回退静态 yaml cn 列表
        # us 宽基：动态 S&P 500 成分股
        if market == "us" and us_benchmark == "sp500":
            sp = sp500_constituents()
            if not sp.empty:
                if demo and "us" in demo_size:
                    sp = sp.head(int(demo_size["us"]))
                for _, r in sp.iterrows():
                    rows.append({"code": r["code"], "name": r.get("name"), "market": "us",
                                 "industry": r.get("industry"), "country": _COUNTRY.get("us")})
                continue
            # 抓取失败 → 回退静态 yaml us 列表
        items = raw.get(market, []) or []
        if demo and market in demo_size:
            items = items[: int(demo_size[market])]
        for it in items:
            rows.append(
                {
                    "code": it["code"],
                    "name": it.get("name"),
                    "market": market,
                    "industry": it.get("industry"),
                    "country": _COUNTRY.get(market),
                }
            )
    return pd.DataFrame(rows)


def universe_to_db(df: pd.DataFrame | None = None) -> int:
    """把股票池写入 stock 表（upsert），返回写入行数。"""
    if df is None:
        df = resolve_universe()
    return upsert_dataframe(df, Stock)


def universe_by_market(df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    df = resolve_universe() if df is None else df
    return {m: g.reset_index(drop=True) for m, g in df.groupby("market")}
