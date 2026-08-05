"""Phase 2：新闻系统。

设计原则（见设计文档第 6 节）：不简单做情感分析，而是**事件抽取**。
模块：
  - sources.py     新闻源（RSS / 公司IR / SEC / 财经API），去重，关联股票
  - embed.py       FAISS + sentence-transformers（本地 qwen3-embedding-0.6B）新闻 embedding
  - llm_events.py  LLM 事件抽取 → 结构化 {event, direction, impact, time_horizon, ...} → news_factor

Phase 1 不执行；接口已留，配齐 settings.news / llm / embedding 后即可启用。
"""
