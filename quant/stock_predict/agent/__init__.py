"""Phase 3：投资分析 Agent（接口预留）。

目标：把「Alpha 预测模型 + LLM 分析」组合成一个能对话、能给理由、能查新闻的 Agent。
Phase 1/2 已具备：预测概率、可解释性（report/explain）、新闻事件（news/llm_events）。
Phase 3 将用 LangChain 把它们串成一个 ReAct 风格的 Agent。

tools（待实现）：
  - get_prediction(code)         查概率/评分/理由
  - get_news_events(code)        查相关新闻事件
  - get_industry_score(industry) 查行业周期
  - run_backtest(config)         跑回测
"""
from __future__ import annotations


def build_agent():  # pragma: no cover - Phase 3
    raise NotImplementedError(
        "Agent (Phase 3) 尚未实现。"
        "可用素材：stock_predict.report.daily / stock_predict.news.llm_events。"
    )
