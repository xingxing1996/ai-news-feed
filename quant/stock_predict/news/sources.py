"""新闻源适配器（设计文档第 6 节）。

分层：
  第一层（免费）：RSS（Reuters/CNBC/Yahoo）、公司IR
  第二层（推荐）：财经API（FMP / Alpha Vantage / Finnhub）
  第三层：公司公告（A股巨潮资讯 / 美股 SEC EDGAR）

统一产出 ``NewsItem``，去重，再关联到股票代码。
所有网络访问均做异常隔离，单源失败不影响其它。
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import pandas as pd

from ..config import get_settings

log = logging.getLogger(__name__)


@dataclass
class NewsItem:
    title: str
    content: str
    publish_time: datetime | None = None
    source: str = ""
    url: str = ""
    related_codes: list[str] = field(default_factory=list)

    def uid(self) -> str:
        h = hashlib.md5((self.title + self.source).encode("utf-8")).hexdigest()
        return h[:16]


# ---------------------------- RSS ---------------------------- #
def fetch_rss(feeds: Iterable[str]) -> list[NewsItem]:
    try:
        import feedparser  # noqa: WPS433
    except Exception:  # noqa: BLE001
        log.warning("[news] feedparser 未安装，跳过 RSS（pip install feedparser）")
        return []
    items: list[NewsItem] = []
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries:
                items.append(
                    NewsItem(
                        title=getattr(e, "title", ""),
                        content=getattr(e, "summary", "") or getattr(e, "description", ""),
                        publish_time=_parse_dt(getattr(e, "published_parsed", None)),
                        source=url,
                        url=getattr(e, "link", ""),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("[news] RSS %s 失败: %s", url, exc)
    return items


# ---------------------------- 财经 API ---------------------------- #
def fetch_fmp(symbols: list[str], key: str) -> list[NewsItem]:
    if not key:
        return []
    import requests  # 局部依赖

    items: list[NewsItem] = []
    for sym in symbols:
        try:
            r = requests.get(
                f"https://financialmodelingprep.com/api/v3/stock_news",
                params={"tickers": sym, "limit": 20, "apikey": key},
                timeout=15,
            )
            r.raise_for_status()
            for e in r.json():
                items.append(
                    NewsItem(
                        title=e.get("title", ""),
                        content=e.get("text", ""),
                        publish_time=_parse_iso(e.get("publishedDate")),
                        source="FMP",
                        url=e.get("url", ""),
                        related_codes=[sym],
                    )
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("[news] FMP %s 失败: %s", sym, exc)
    return items


def fetch_finnhub(symbols: list[str], key: str) -> list[NewsItem]:
    if not key:
        return []
    import requests

    items: list[NewsItem] = []
    today = datetime.today().strftime("%Y-%m-%d")
    for sym in symbols:
        try:
            r = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": sym, "from": today, "to": today, "token": key},
                timeout=15,
            )
            r.raise_for_status()
            for e in r.json():
                items.append(
                    NewsItem(
                        title=e.get("headline", ""),
                        content=e.get("summary", ""),
                        publish_time=datetime.fromtimestamp(e.get("datetime", 0)) if e.get("datetime") else None,
                        source="Finnhub",
                        url=e.get("url", ""),
                        related_codes=[sym],
                    )
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("[news] Finnhub %s 失败: %s", sym, exc)
    return items


# ---------------------------- A股个股新闻（AKShare）----------------------------- #
def fetch_akshare_news(codes: list[str], limit: int = 15) -> list[NewsItem]:
    """A股个股新闻（AKShare ``stock_news_em``）。codes 为带后缀代码（600519.SH）。"""
    try:
        import akshare as ak  # noqa: WPS433
    except Exception:  # noqa: BLE001
        log.warning("[news] akshare 未安装，跳过 A股个股新闻")
        return []
    items: list[NewsItem] = []
    for code in codes:
        sym = code.split(".")[0]
        try:
            df = ak.stock_news_em(symbol=sym)
        except Exception as exc:  # noqa: BLE001
            log.debug("[news] AKShare %s 失败: %s", code, exc)
            continue
        if df is None or df.empty:
            continue
        for _, r in df.head(limit).iterrows():
            items.append(
                NewsItem(
                    title=str(r.get("新闻标题", "")),
                    content=str(r.get("新闻内容", "")),
                    publish_time=_parse_iso(str(r.get("发布时间", ""))),
                    source=str(r.get("文章来源", "akshare")),
                    url=str(r.get("新闻链接", "")),
                    related_codes=[code],
                )
            )
    return items


# ---------------------------- 去重 / 关联 ---------------------------- #
def dedup(items: list[NewsItem]) -> list[NewsItem]:
    seen, out = set(), []
    for it in items:
        if it.uid() not in seen:
            seen.add(it.uid())
            out.append(it)
    return out


def associate(items: list[NewsItem], universe: pd.DataFrame) -> list[NewsItem]:
    """按股票名称/代码做关键词关联（简单版；Phase 2 可换成向量检索）。"""
    name_map = {str(r.name).lower(): r.code for r in universe.itertuples(index=False)}
    code_map = {str(r.code).lower(): r.code for r in universe.itertuples(index=False)}
    for it in items:
        text = (it.title + " " + it.content).lower()
        codes = set(it.related_codes)
        for k, code in {**name_map, **code_map}.items():
            if k and k in text:
                codes.add(code)
        it.related_codes = sorted(codes)
    return items


def collect(universe: pd.DataFrame | None = None) -> list[NewsItem]:
    """按 settings.news.sources 收集并去重、关联股票。"""
    cfg = get_settings()
    ncfg = cfg.get("news", {})
    symbols = (universe["code"].tolist() if universe is not None else [])
    items: list[NewsItem] = []
    for src in ncfg.get("sources", []):
        if src == "rss":
            items += fetch_rss(ncfg.get("rss_feeds", []))
        elif src == "akshare":
            cn_codes = universe[universe["market"] == "cn"]["code"].tolist() if universe is not None else []
            items += fetch_akshare_news(cn_codes)
        elif src == "sec":
            from .sec_edgar import fetch_sec_filings  # 美股 SEC 公告（免费/稳定）

            us_codes = universe[universe["market"] == "us"]["code"].tolist() if universe is not None else []
            items += fetch_sec_filings(us_codes)
        elif src == "fmp":
            items += fetch_fmp(symbols, cfg.data.get("fmp_key", ""))
        elif src == "finnhub":
            items += fetch_finnhub(symbols, cfg.data.get("finnhub_key", ""))
    if ncfg.get("dedup", True):
        items = dedup(items)
    if universe is not None:
        items = associate(items, universe)
    log.info("[news] 收集到 %d 条新闻", len(items))
    return items


def _parse_dt(tp):
    if not tp:
        return None
    try:
        import time as _t

        return datetime.fromtimestamp(_t.mktime(tp))
    except Exception:  # noqa: BLE001
        return None


def _parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None
