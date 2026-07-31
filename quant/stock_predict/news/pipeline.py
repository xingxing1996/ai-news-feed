"""新闻流水线编排：采集 → 入库 → DeepSeek 事件抽取 → news_score / 事件清单。

输出：
  data/warehouse/news_events.parquet  每只股票最近事件 + 综合 news_score（供日报引用）
  news 表（SQLite）                     原始新闻

成本控制：LLM 调用按 ``news.max_codes`` / ``news.per_stock`` 截断。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import pandas as pd

from ..config import get_settings
from ..data.models import News
from ..data.universe import resolve_universe
from ..data.warehouse import upsert_dataframe, write_parquet
from . import sources
from .llm_events import events_to_score, extract_events_batch

log = logging.getLogger(__name__)


def run_news_pipeline(universe: pd.DataFrame | None = None, max_codes: int | None = None,
                      per_stock: int | None = None) -> dict:
    cfg = get_settings()
    ncfg = cfg.get("news", {})
    universe = resolve_universe() if universe is None else universe
    max_codes = max_codes or int(ncfg.get("max_codes", 20))
    per_stock = per_stock or int(ncfg.get("per_stock", 3))

    # 1) 采集 + 关联
    items = sources.collect(universe)
    log.info("[news] 采集到 %d 条新闻", len(items))

    # 2) 原始入库（news 表）
    if items:
        rows = [
            {
                "stock": ",".join(it.related_codes),
                "title": it.title,
                "content": (it.content or "")[:2000],
                "publish_time": it.publish_time or datetime.now(),
                "source": it.source,
            }
            for it in items
        ]
        upsert_dataframe(pd.DataFrame(rows), News)

    # 3) 按股票分组，取最近 per_stock 条做事件抽取（截断到 max_codes 控成本）
    by_code: dict[str, list] = {}
    for it in items:
        for code in it.related_codes:
            by_code.setdefault(code, []).append(it)

    # 优先保留 universe 靠前的股票
    ordered = [c for c in universe["code"].tolist() if c in by_code][:max_codes]

    # 批量抽取（一次 LLM 处理多条新闻，降 RPM 压力）
    pairs: list[tuple[str, list[str]]] = []
    code_its: dict[str, list] = {}
    for code in ordered:
        its = sorted(by_code[code], key=lambda x: x.publish_time or datetime.min, reverse=True)[:per_stock]
        code_its[code] = its
        for it in its:
            text = (it.title + "。 " + it.content)[:1200]
            pairs.append((text, [code]))

    bsize = int(ncfg.get("batch_size", 3))
    batched = extract_events_batch(pairs, batch_size=bsize) if pairs else {}
    n_calls = max(1, (len(pairs) + bsize - 1) // bsize)

    records = []
    for code in ordered:
        events = batched.get(code, [])
        score = events_to_score(events)
        records.append(
            {
                "code": code,
                "news_score": score,
                "n_news": len(code_its.get(code, [])),
                "events": json.dumps(events, ensure_ascii=False),
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
        )

    out = pd.DataFrame(records)
    if not out.empty:
        write_parquet(out, "news_events")
    log.info("[news] 事件抽取完成: %d 只股票, %d 次 LLM 调用", len(out), n_calls)
    return {"n_items": len(items), "n_codes": len(out), "n_llm_calls": n_calls}


def load_news_events() -> pd.DataFrame:
    from ..data.warehouse import read_parquet

    return read_parquet("news_events")
