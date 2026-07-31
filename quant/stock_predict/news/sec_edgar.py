"""美股 SEC EDGAR 公告抓取（免费、稳定、无需 API key）。

SEC 的 8-K/10-Q/10-K 是**最干净的事件源**（一手、结构化、官方），
比二手新闻更适合 LLM 事件抽取（设计文档：「财报文本 > 新闻」）。

流程：ticker → CIK（SEC 官方映射表）→ submissions JSON → 近期 filings → NewsItem。
限速 10 req/s（SEC 规定），本模块按 max_codes 截断。
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from ..config import get_settings
from .sources import NewsItem

log = logging.getLogger(__name__)

_USER_AGENT = "stock-predict research bot (quant-research@example.com)"
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_TICKER_CACHE: Path | None = None
_CIK_MAP: dict[str, str] | None = None  # ticker(upper) -> CIK


def _cache_path() -> Path:
    global _TICKER_CACHE
    if _TICKER_CACHE is None:
        _TICKER_CACHE = Path(get_settings().paths.raw_dir) / "sec_tickers.json"
    return _TICKER_CACHE


def _get(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except Exception as exc:  # noqa: BLE001
        log.debug("[sec] 请求失败 %s: %s", url, exc)
        return None


def _load_cik_map() -> dict[str, str]:
    """加载/缓存 SEC ticker→CIK 映射。"""
    global _CIK_MAP
    if _CIK_MAP is not None:
        return _CIK_MAP
    path = _cache_path()
    data = None
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            data = None
    if not data:
        data = _get(_TICKER_MAP_URL)
        if data:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
    m: dict[str, str] = {}
    if isinstance(data, dict):
        for v in data.values():
            tkr = str(v.get("ticker", "")).upper()
            cik = str(v.get("cik_str", "")).zfill(10)
            if tkr and cik:
                m[tkr] = cik
    _CIK_MAP = m
    return m


def _recent_filings(cik: str, limit: int = 8) -> list[dict]:
    """取该 CIK 最近 limit 条 8-K/10-Q/10-K/6-K filings。"""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = _get(url)
    if not data:
        return []
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    items = recent.get("items", [])
    descs = recent.get("primaryDocDescription", [])
    out = []
    for i, form in enumerate(forms):
        if form in ("8-K", "10-Q", "10-K", "6-K", "20-F"):
            try:
                dt = datetime.fromisoformat(dates[i])
            except (IndexError, ValueError):
                dt = None
            out.append(
                {
                    "form": form,
                    "date": dt,
                    "items": items[i] if i < len(items) else "",
                    "desc": descs[i] if i < len(descs) else "",
                }
            )
        if len(out) >= limit:
            break
    return out


def fetch_sec_filings(codes: list[str], limit: int = 8) -> list[NewsItem]:
    """codes 为美股 ticker（AAPL）。返回近期 SEC 公告 NewsItem。"""
    cik_map = _load_cik_map()
    if not cik_map:
        log.warning("[sec] 未取到 SEC ticker→CIK 映射（网络？），跳过 SEC")
        return []
    items: list[NewsItem] = []
    for code in codes:
        tkr = code.split(".")[0].upper()
        cik = cik_map.get(tkr)
        if not cik:
            continue
        for f in _recent_filings(cik, limit):
            ev_desc = _form_text(f["form"], f["items"], f["desc"])
            items.append(
                NewsItem(
                    title=f"{tkr} {f['form']} filing",
                    content=ev_desc,
                    publish_time=f["date"],
                    source="SEC",
                    related_codes=[code],
                )
            )
        time.sleep(0.12)  # 守 SEC 10 req/s
    log.info("[sec] 取到 %d 条 SEC 公告（%d 只）", len(items), len(codes))
    return items


_8K_ITEMS = {
    "1.01": "entered a definitive material agreement",
    "1.02": "terminated a material agreement",
    "2.01": "completed an acquisition or asset sale",
    "2.02": "released earnings / results of operations",
    "2.03": "created a direct financial obligation (debt)",
    "2.04": "accelerated a financial obligation",
    "2.05": "announced costs associated with exit or disposal (restructuring/layoffs)",
    "3.01": "received a delisting notice",
    "3.02": "completed unregistered sale of equity (dilution)",
    "3.03": "material modification to security holders' rights",
    "4.01": "change in certifying accountant",
    "5.01": "election of a new director",
    "5.02": "departure of a director or officer",
    "5.03": "amendment to charter or bylaws",
    "5.07": "matters submitted to a shareholder vote",
    "7.01": "Regulation FD disclosure (material info shared)",
    "8.01": "other material event",
    "9.01": "financial statements and exhibits (boilerplate)",
}


def _form_text(form: str, items: str, desc: str) -> str:
    """把 filing 元数据拼成 LLM 可读文本（解码 8-K item 编号）。"""
    meaning = {
        "8-K": "material event filing (8-K)",
        "10-Q": "quarterly report (10-Q)",
        "10-K": "annual report (10-K)",
        "6-K": "foreign private issuer report (6-K)",
        "20-F": "foreign annual report (20-F)",
    }.get(form, form)
    parts = [f"Company filed {meaning}."]
    if items and form == "8-K":
        decoded = []
        for code in [c.strip() for c in items.split(",") if c.strip()]:
            if code in _8K_ITEMS:
                decoded.append(f"Item {code}: {_8K_ITEMS[code]}")
        if decoded:
            parts.append("; ".join(decoded) + ".")
    elif items:
        parts.append(f"Items: {items}.")
    if desc:
        parts.append(desc)
    return " ".join(parts)
