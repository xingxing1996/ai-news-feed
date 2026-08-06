"""LLM 事件抽取（设计文档第 6 节，真正有价值的部分）。

不是「正面/负面」情感，而是结构化事件：
  {
    "company": "SK Hynix",
    "event": "HBM demand increase",
    "direction": "positive",        # positive / negative / neutral
    "impact": 0.85,                 # 0~1
    "time_horizon": "6 months",     # days / weeks / months / quarters
    "confidence": 0.9,
    "related_company": ["Nvidia", "Micron"]
  }
聚合为 ``news_score`` 进入特征。

LLM 客户端走 OpenAI 兼容接口（你的 model_provider / Ollama / vLLM），
配置见 settings.llm。无 LLM 时退化为关键词规则抽取（保证可用）。
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time

from ..config import get_settings

log = logging.getLogger(__name__)

# ---- 火山方舟 Ark（OpenAI 兼容）+ 模型轮询池（从 config/models.yaml 读，不再写死）----
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


def _load_pool() -> list[str]:
    """从 config/models.yaml 读具体版本 id；读不到则兜底。"""
    try:
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[3]  # quant/stock_predict/news/ -> 仓库根
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from common.models import get_model_pool

        pool = get_model_pool()
        if pool:
            return pool
    except Exception:
        pass
    return ["glm-5-2-260617", "doubao-seed-2-1-pro-260628"]  # 兜底（deepseek-v3-2 已下掉）


MODEL_POOL = _load_pool()
DEFAULT_MODEL = MODEL_POOL[0] if MODEL_POOL else "glm-5-2-260617"


def _pool_order() -> list[str]:
    """完全随机打乱模型顺序（不每次都从第一个开始）。"""
    order = list(MODEL_POOL)
    random.shuffle(order)
    return order

_SYSTEM = (
    "你是金融事件抽取助手。从新闻中抽取结构化事件，严格输出 JSON：{\"events\":[...]}，"
    "每个事件字段：company(主体公司), event(一句话事件摘要，≤20字), "
    "direction(positive/negative/neutral), impact(0~1 影响幅度), "
    "time_horizon(days/weeks/months/quarters), confidence(0~1), "
    "related_company(相关公司数组)。"
    "结合行业上下文判断方向：例如半导体/HBM「增产/需求增/订单增」通常 positive；"
    "普通商品「库存增加/降价」通常 negative；「减持/违规/停产/退市」negative；"
    "「回购/分红/获批/超预期」positive。不确定就 neutral。"
)

_POSITIVE = ["增长", "超预期", "突破", "订单", "需求增加", "回升", "获批", "回购", "分红", "利好", "increase", "beat", "surge", "win", "approved"]
_NEGATIVE = ["下滑", "亏损", "减持", "违规", "退市", "警告", "停产", "库存增加", "下滑", "lawsuit", "miss", "drop", "recall", "downgrade"]


def _llm_client():
    """OpenAI 兼容客户端，指向火山方舟 Ark。api_key 优先环境变量 ARK_API_KEY。"""
    cfg = get_settings().get("llm", {})
    api_key = os.getenv("ARK_API_KEY") or cfg.get("api_key")
    base_url = cfg.get("base_url") or ARK_BASE_URL
    if not api_key:
        log.warning("[news] 未配置 ARK_API_KEY（env 或 settings.llm.api_key），LLM 不可用，走规则抽取")
        return None, base_url
    try:
        from openai import OpenAI  # noqa: WPS433

        return OpenAI(base_url=base_url, api_key=api_key, max_retries=0), base_url
    except Exception:  # noqa: BLE001
        log.info("[news] openai 库未安装，使用规则抽取")
        return None, base_url


def extract_events(text: str, related_codes: list[str] | None = None) -> list[dict]:
    client, base_url = _llm_client()
    cfg = get_settings().get("llm", {})
    model = cfg.get("model") or DEFAULT_MODEL
    if client is not None:
        try:
            kwargs = dict(
                model=model,
                temperature=float(cfg.get("temperature", 0.1)),
                timeout=float(cfg.get("timeout", 60)),
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": text[:4000]},
                ],
            )
            resp = _create_retry(client, kwargs)
            content = resp.choices[0].message.content or "{}"
            data = _safe_json(content)
            evs = data.get("events") if isinstance(data, dict) else data
            if isinstance(evs, list):
                return [_norm_event(e, related_codes or []) for e in evs if isinstance(e, dict)]
        except Exception as exc:  # noqa: BLE001
            log.warning("[news] LLM 抽取失败，退化规则：%s", exc)
    return _rule_extract(text, related_codes or [])


def _create_retry(client, kwargs, max_tries: int = 3):
    """串行发 chat 请求：随机挑模型，429/超时切下一个，最多尝试 max_tries 次。

    - 模型顺序每次随机打乱（不每次都默认第一个）；
    - 单模型 429/超时 → 立即切下一个（failover，不等待）；
    - 其它错误直接抛；超过 max_tries（默认3）仍失败 → 抛给上层退化规则。
    """
    base = dict(kwargs)
    base.pop("response_format", None)
    order = _pool_order()[:max_tries]
    last_exc = None
    for i, model in enumerate(order):
        try:
            k = dict(base)
            k["model"] = model
            return client.chat.completions.create(**k)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            sl = str(exc).lower()
            if "403" in sl or "forbidden" in sl or "401" in sl or "unauthorized" in sl:
                log.warning("[news] 火山方舟 API Key 无权限或未开启模型权限 (403/401)，熔断重试，退化规则新闻抽取")
                break
            if ("429" in sl or "ratelimit" in sl or "toomany" in sl
                    or "timed out" in sl or "timeout" in sl or "apitimeout" in sl
                    or "404" in sl or "notfound" in sl or "does not exist" in sl
                    or "no access" in sl):
                log.warning("[news] %s 不可用(限流/超时/404)，换下一个模型（%d/%d）", model, i + 1, len(order))
                continue
            raise  # 其它错误直接抛
    raise last_exc if last_exc else RuntimeError("LLM 模型池全部失败")


def extract_events_batch(pairs: list[tuple[str, list[str]]], batch_size: int = 3) -> dict[str, list[dict]]:
    """批量事件抽取：把多条新闻一次性发给 LLM，降低 RPM 压力。

    pairs: [(text, [code]), ...]。返回 {code: [events]}。失败项退化规则。
    """
    client, _ = _llm_client()
    cfg = get_settings().get("llm", {})
    model = cfg.get("model") or DEFAULT_MODEL
    out: dict[str, list[dict]] = {}

    def _flush(batch):
        if client is None:
            for txt, codes in batch:
                for c in codes:
                    out.setdefault(c, []).extend(_rule_extract(txt, codes))
            return
        numbered = "\n".join(f"[{i}] {txt[:600]}" for i, (txt, _) in enumerate(batch))
        prompt = (
            "对以下每条新闻分别抽取事件，输出 JSON：{\"results\":[{\"idx\":0,\"events\":[...]}]}。"
            "事件字段同前(company/event/direction/impact/time_horizon/confidence/related_company)。"
            "无事件的条目 events 为空数组。\n\n" + numbered
        )
        try:
            resp = _create_retry(
                client,
                dict(model=model, temperature=float(cfg.get("temperature", 0.1)),
                     timeout=float(cfg.get("timeout", 60)),
                     messages=[{"role": "system", "content": _SYSTEM},
                               {"role": "user", "content": prompt}]),
            )
            data = _safe_json(resp.choices[0].message.content or "{}") or {}
            res_map = {r.get("idx"): r.get("events", []) for r in (data.get("results") or [])}
        except Exception as exc:  # noqa: BLE001
            log.warning("[news] 批量抽取失败，退化规则：%s", exc)
            res_map = None
        for i, (txt, codes) in enumerate(batch):
            evs = []
            if res_map is not None and i in res_map:
                evs = [_norm_event(e, codes) for e in res_map[i] if isinstance(e, dict)]
            elif res_map is None:
                evs = _rule_extract(txt, codes)
            for c in codes:
                out.setdefault(c, []).extend(evs)

    batch = []
    for txt, codes in pairs:
        batch.append((txt, codes))
        if len(batch) >= batch_size:
            _flush(batch)
            batch = []
            time.sleep(1.5)  # 节流，避免 RPM 超限
    if batch:
        _flush(batch)
    return out


def _norm_event(e: dict, related_codes: list[str]) -> dict:
    """补全/规范事件字段。"""
    e.setdefault("company", related_codes[0] if related_codes else "UNKNOWN")
    e.setdefault("event", "")
    e["direction"] = str(e.get("direction", "neutral")).lower()
    if e["direction"] not in ("positive", "negative", "neutral"):
        e["direction"] = "neutral"
    try:
        e["impact"] = max(0.0, min(1.0, float(e.get("impact", 0.5))))
    except (TypeError, ValueError):
        e["impact"] = 0.5
    try:
        e["confidence"] = max(0.0, min(1.0, float(e.get("confidence", 0.5))))
    except (TypeError, ValueError):
        e["confidence"] = 0.5
    e.setdefault("time_horizon", "weeks")
    rc = e.get("related_company") or []
    if related_codes:
        rc = list({*rc, *related_codes})
    e["related_company"] = rc
    return e


def _rule_extract(text: str, related_codes: list[str]) -> list[dict]:
    low = text.lower()
    pos = sum(k in low for k in _POSITIVE)
    neg = sum(k in low for k in _NEGATIVE)
    if pos == 0 and neg == 0:
        return []
    direction = "positive" if pos >= neg else "negative"
    impact = min(0.5 + 0.1 * max(pos, neg), 0.95)
    codes = related_codes or ["UNKNOWN"]
    return [
        {
            "company": codes[0],
            "event": text[:60],
            "direction": direction,
            "impact": round(impact, 2),
            "time_horizon": "weeks",
            "confidence": 0.5,
            "related_company": codes[1:],
        }
    ]


def _safe_json(s: str):
    m = re.search(r"\[.*\]|\{.*\}", s, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def events_to_score(events: list[dict]) -> float:
    """把事件聚合成单个 news_score ∈ [0,1]，方向加权。"""
    if not events:
        return 0.5
    score = 0.0
    wsum = 0.0
    for e in events:
        w = float(e.get("confidence", 0.5)) * float(e.get("impact", 0.5))
        sign = {"positive": 1.0, "negative": -1.0}.get(e.get("direction"), 0.0)
        score += sign * w
        wsum += w
    if wsum == 0:
        return 0.5
    return float(max(0.0, min(1.0, 0.5 + 0.5 * (score / wsum))))


def news_factor_for_items(items: list[dict]) -> dict[str, float]:
    """items: [{text, related_codes}] → {code: news_score}。"""
    out: dict[str, float] = {}
    for it in items:
        evs = extract_events(it.get("text", ""), it.get("related_codes", []))
        s = events_to_score(evs)
        for code in it.get("related_codes", ["UNKNOWN"]):
            out[code] = max(out.get(code, 0.0), s)
    return out
