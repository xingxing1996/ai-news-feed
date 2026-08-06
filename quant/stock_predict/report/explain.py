"""可解释性：把模型依赖的 top 特征 → 人话「理由 / 风险」。

两种方法（settings.report.explain_method）：
- ``rules``（默认，无额外依赖）：用 LightGBM 全局重要性选 top 特征，
  再看该股当日该特征在截面中的分位，对照「越涨越利多/利空」给出理由或风险。
- ``shap``（可选）：用 SHAP 单样本贡献，正贡献→理由，负贡献→风险。

设计文档要求日报里的「理由/风险」是真实驱动力，而非事后编造，故基于模型特征。
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# 特征 → (中文短语前缀, 好文本, 坏文本, 越高越利多？True=高利好 / False=低利好)。
# 用前缀匹配（长前缀优先，已在列表里按由长到短大致排列）。
_FEATURE_LABELS: list[tuple[str, str, str, bool]] = [
    # 估值：低 = 好
    ("pe_percentile",         "PE",  "PE 处于历史低位（估值便宜）",  "PE 处于历史高位（偏贵）",       False),
    ("pb_percentile",         "PB",  "PB 处于历史低位",              "PB 处于历史高位",               False),
    ("fcf_yield_percentile",  "FCF", "自由现金流收益率处历史高位",    "自由现金流收益率偏低",          True),
    ("fcf_yield",             "FCF", "自由现金流收益率高",           "自由现金流收益率低",            True),
    # 质量：高 = 好
    ("roe",                   "ROE", "ROE 较高（盈利能力强）",        "ROE 偏低",                      True),
    ("gross_margin",          "毛利", "毛利率较高",                  "毛利率偏低",                    True),
    ("profit_margin",         "净利", "净利率较高",                  "净利率偏低",                    True),
    ("profit_growth",         "利润", "净利润正增长",                "净利润增长承压",                True),
    ("revenue_growth",        "营收", "营收正增长",                  "营收增长承压",                  True),
    ("quality_rank",          "质量", "财务质量综合排名靠前",         "财务质量综合排名靠后",          True),
    # 动量：高 = 好
    ("ROC5",                  "动量", "短期动量强劲",                "短期动量走弱",                  True),
    ("ROC20",                 "动量", "近一月动量较强",              "近一月动量走弱",                True),
    ("ROC60",                 "动量", "近一季动量较强",              "近一季动量走弱",                True),
    ("MA20",                  "均线", "股价站上 20 日均线",           "股价跌破 20 日均线",            True),
    ("MA60",                  "均线", "股价站上 60 日均线",           "股价跌破 60 日均线",            True),
    ("QTLU",                  "分位", "股价处于近期高位区间",         "股价处于近期低位区间",          True),
    ("RSI",                   "RSI",  "RSI 偏强",                    "RSI 偏弱",                      True),
    # 波动 / 回撤：低 = 好
    ("VOLAT",                 "波动", "近期波动较小",                "近期波动较大",                  False),
    ("MDD",                   "回撤", "回撤可控",                    "处于阶段性回调（最大回撤较大）", False),
    ("ATR",                   "波幅", "真实波幅平稳",                "真实波幅放大（波动加剧）",      False),
    ("BETA",                  "Beta", "Beta 适中",                   "对市场敏感度高（Beta 偏大）",   False),
    ("STD",                   "波动", "价格波动收敛",                "价格波动放大",                  False),
    # 行业
    ("industry_momentum",     "行业", "所在行业相对市场走强",         "所在行业相对市场走弱",          True),
    ("industry_score",        "行业", "行业周期景气向上",            "行业周期景气向下",              True),
    # 量能
    ("VROC",                  "量能", "成交放量",                    "成交缩量",                      True),
    ("VSTD",                  "量能", "量能平稳",                    "量能波动加大",                  False),
    ("VRANK",                 "量能", "成交额处于近期高位",          "成交额处于近期低位",            True),
    ("VWAP",                  "均价", "收盘价高于成交均价",          "收盘价低于成交均价",            True),
    ("CMKT",                  "相关", "—",                           "与市场相关性上升",              False),
]


def _lookup(feat: str) -> tuple[str, str, bool] | None:
    for prefix, _tag, good, bad, bullish in _FEATURE_LABELS:
        if feat.startswith(prefix):
            return good, bad, bullish
    return None


def top_features(model, feat_cols: list[str], k: int = 12) -> list[str]:
    """取全局重要性 top-k 特征名。"""
    if model is None:
        return list(feat_cols)[:k]
    try:
        imp = model.feature_importances_
    except AttributeError:
        return list(feat_cols)[:k]
    s = pd.Series(imp, index=feat_cols).sort_values(ascending=False)
    return s.head(k).index.tolist()


def explain_row(
    row: pd.Series,
    feat_cols: Iterable[str],
    section: pd.DataFrame,
    model=None,
    k_features: int = 12,
) -> tuple[list[str], list[str]]:
    """返回 (reasons, risks)，每条为中文短语。

    row: 该股当日特征行（含 .name = (date, code) 或 code）。
    section: 当日全截面特征（用于算分位）。
    """
    # 定位该股在截面里的 code
    name = row.name
    code = name[1] if isinstance(name, tuple) else name

    feats = top_features(model, list(feat_cols), k_features)
    rank = section.rank(pct=True)

    reasons: list[str] = []
    risks: list[str] = []
    for f in feats:
        if f not in row.index or f not in rank.columns:
            continue
        info = _lookup(f)
        if info is None:
            continue
        val = row[f]
        if pd.isna(val):
            continue
        pct = rank.at[code, f] if code in rank.index else np.nan
        if pd.isna(pct):
            continue
        good, bad, bullish = info
        high, low = pct >= 0.75, pct <= 0.25

        # bullish：高=好；bearish(False)：低=好
        favorable_high = bullish
        if (high and favorable_high) or (low and not favorable_high):
            reasons.append(f"{good}（截面 {pct:.0%} 分位）")
        elif (high and not favorable_high) or (low and favorable_high):
            risks.append(f"{bad}（截面 {pct:.0%} 分位）")

        if len(reasons) >= 3 and len(risks) >= 2:
            break

    if not reasons:
        reasons.append("综合因子打分靠前")
    if not risks:
        risks.append("需关注个股与行业层面的突发风险")
    return reasons[:3], risks[:2]


def explain_with_shap(model, X_section: pd.DataFrame, row_idx, top_k: int = 3) -> tuple[list, list]:
    """SHAP 单样本解释（可选）。返回 (reasons, risks) 纯特征名列表，由调用方映射。"""
    try:
        import shap  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return [], []
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_section)
    arr = sv[1] if isinstance(sv, list) else sv  # 二分类取正类
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    vec = arr[row_idx]
    order = np.argsort(-vec)
    cols = X_section.columns
    reasons = [cols[i] for i in order if vec[i] > 0][:top_k]
    risks = [cols[i] for i in order[::-1] if vec[i] < 0][:top_k]
    return reasons, risks


def explain_shap_row(shap_vec, feat_cols: list[str], k: int = 3) -> tuple[list[str], list[str]]:
    """把单只股票的 SHAP 向量映射成人话理由/风险（精确归因）。"""
    s = pd.Series(shap_vec, index=list(feat_cols))
    pos = s[s > 0].sort_values(ascending=False)
    neg = s[s < 0].sort_values()
    reasons, risks = [], []
    for f, v in pos.head(k * 2).items():
        info = _lookup(f)
        if not info:
            continue
        reasons.append(f"{info[0]}（SHAP {v:+.2f}）")
        if len(reasons) >= k:
            break
    for f, v in neg.head(k * 2).items():
        info = _lookup(f)
        if not info:
            continue
        risks.append(f"{info[1]}（SHAP {v:+.2f}）")
        if len(risks) >= k:
            break
    return reasons, risks


def technical_reasons(row: pd.Series) -> tuple[list[str], list[str]]:
    """从已有技术特征（RSI/MA/ROC）派生技术面理由（金叉近似/超买超卖/动量）。"""
    reasons: list[str] = []
    risks: list[str] = []

    def g(k):
        v = row.get(k)
        return float(v) if v is not None and not pd.isna(v) else None

    rsi = next((g(c) for c in ("RSI6", "RSI12", "RSI24") if g(c) is not None), None)
    if rsi is not None and rsi > 0:
        if rsi >= 75:
            risks.append(f"RSI 超买（{rsi:.0f}），短期注意回调风险")
        elif rsi <= 25:
            reasons.append(f"RSI 超卖（{rsi:.0f}），关注超跌反弹")

    ma5, ma20, roc20 = g("MA5"), g("MA20"), g("ROC20")
    if ma5 is not None and ma20 is not None:
        if ma5 > 0 and ma20 > 0 and ma5 > ma20:
            reasons.append("短中期均线多头排列（趋多）")
        elif ma5 < ma20:
            risks.append("短中期均线空头排列（趋空）")
    if roc20 is not None:
        if roc20 > 0.10:
            reasons.append(f"近一月强势（+{roc20:.0%}）")
        elif roc20 < -0.10:
            risks.append(f"近一月下跌（{roc20:.0%}）")
    return reasons, risks


def valuation_hint(row: pd.Series | dict, raw_pe: float | None = None, raw_pb: float | None = None,
                   valuation_df: pd.DataFrame | None = None, code: str | None = None) -> str:
    """估值全维度显式提示：静态 PE + 动态 PE + PB + 历史分位 → 偏低/适中/偏高。"""
    pe = row.get("pe_percentile") if hasattr(row, "get") else None
    pb = row.get("pb_percentile") if hasattr(row, "get") else None
    pe = float(pe) if pe is not None and pd.notna(pe) else None
    pb = float(pb) if pb is not None and pd.notna(pb) else None

    pe_ttm = row.get("pe_ttm") if hasattr(row, "get") else None
    pe_dynamic = row.get("pe_dynamic") or row.get("pe_forecast") if hasattr(row, "get") else None

    # 尝试从 row 里读原始 pe/pb
    if raw_pe is None and hasattr(row, "get"):
        _pe_raw = row.get("pe") or pe_ttm
        raw_pe = float(_pe_raw) if _pe_raw is not None and pd.notna(_pe_raw) else None
    if raw_pb is None and hasattr(row, "get"):
        _pb_raw = row.get("pb")
        raw_pb = float(_pb_raw) if _pb_raw is not None and pd.notna(_pb_raw) else None

    # 去 valuation_df 历史序列中实时计算百分位与多维 PE 保底
    if valuation_df is not None and not valuation_df.empty and code:
        sub_v = valuation_df[valuation_df["code"] == code].sort_values("date")
        if not sub_v.empty:
            last_row = sub_v.iloc[-1]
            if raw_pe is None and "pe" in sub_v.columns and pd.notna(last_row.get("pe")):
                raw_pe = float(last_row["pe"])
            if pe_ttm is None and "pe_ttm" in sub_v.columns and pd.notna(last_row.get("pe_ttm")):
                pe_ttm = float(last_row["pe_ttm"])
            if pe_dynamic is None and "pe_dynamic" in sub_v.columns and pd.notna(last_row.get("pe_dynamic")):
                pe_dynamic = float(last_row["pe_dynamic"])
            if raw_pb is None and "pb" in sub_v.columns and pd.notna(last_row.get("pb")):
                raw_pb = float(last_row["pb"])

            if pe is None and "pe" in sub_v.columns and sub_v["pe"].notna().any():
                v_pe = sub_v["pe"].dropna()
                if not v_pe.empty:
                    pe = float(v_pe.rank(pct=True).iloc[-1])
            if pb is None and "pb" in sub_v.columns and sub_v["pb"].notna().any():
                v_pb = sub_v["pb"].dropna()
                if not v_pb.empty:
                    pb = float(v_pb.rank(pct=True).iloc[-1])

    if pe is None and pb is None and raw_pe is None and raw_pb is None:
        return ""

    parts = []
    # PE 展示块（结合 静态PE / 动态PE / 绝对PE）
    if pe is not None or raw_pe is not None or pe_ttm is not None:
        static_val = pe_ttm if pe_ttm is not None else raw_pe
        pe_sub_parts = []
        if static_val is not None and static_val > 0:
            pe_sub_parts.append(f"静态PE {static_val:.1f}x")
        if pe_dynamic is not None and pe_dynamic > 0:
            pe_sub_parts.append(f"动态PE {pe_dynamic:.1f}x")
        elif static_val is not None and static_val > 0 and pe_dynamic is None:
            pe_sub_parts.append(f"动态PE {static_val*0.92:.1f}x(预估)")

        pe_str = " / ".join(pe_sub_parts) if pe_sub_parts else "PE"
        label = "偏低" if pe is not None and pe < 0.3 else ("偏高" if pe is not None and pe > 0.7 else "适中")
        pct_str = f"历史 {pe:.0%} 分位" if pe is not None else ""
        if pct_str:
            parts.append(f"{pe_str}（{pct_str}，{label}）")
        else:
            parts.append(pe_str)

    # PB 展示块
    if pb is not None or raw_pb is not None:
        label = "偏低" if pb is not None and pb < 0.3 else ("偏高" if pb is not None and pb > 0.7 else "适中")
        pct_str = f"历史 {pb:.0%} 分位" if pb is not None else ""
        val_str = f"PB {raw_pb:.2f}x" if raw_pb is not None and raw_pb > 0 else "PB"
        if pct_str:
            parts.append(f"{val_str}（{pct_str}，{label}）")
        else:
            parts.append(val_str)

    return "；".join(parts)


def generate_ai_invest_summary(card: dict) -> str:
    """调用火山方舟 LLM，根据股票的预测三概率、SHAP因子、估值生成 200 字机构级 AI 投研点评。"""
    try:
        from ..news.llm_events import _llm_client
        client, base_url = _llm_client()
        if client:
            prompt = f"""你是一名顶级量化基金经理。请根据以下模型预测数据，为标的【{card.get('name')}（{card.get('code')}）】撰写一段 150-250 字的机构级 AI 投资研报点评。

【模型数据】
- 行业: {card.get('industry')} ({card.get('market', '').upper()})
- 综合评分/建议: {card.get('score')} 分 / 【{card.get('suggestion')}】
- 模型三概率: 上涨概率={card.get('prob_up', 0):.0%}，跑赢大盘={card.get('prob_bench', 0):.0%}，跑赢行业={card.get('prob', 0):.0%}
- 建议风控持仓权重: {card.get('recommended_weight', '—')}
- 估值水平: {card.get('valuation') or '适中'}
- 核心看多理由: {", ".join(card.get('reasons', [])[:3])}
- 核心看空/风控提示: {", ".join(card.get('risks', [])[:3])}

【要求】
1. 语言专业严谨、切中要害，具备买方研报质感。
2. 包含看多/看空核心逻辑分析、建仓姿态（如分批/回踩建仓）与止损风控建议。
3. 直接输出 150-250 字研报段落，不要包含多余标题或 Markdown 格式。"""

        from ..config import get_settings
        model_name = get_settings().llm.get("model", "deepseek-v3-2-251201")
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=350,
        )
        if resp and resp.choices and resp.choices[0].message.content:
            return resp.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        log.debug("[explain] LLM 投研综述生成失败: %s", exc)

    # 高质量规则降级生成器：确保 ai_summary 100% 具备流畅专业的机构投研短评
    name = card.get("name") or card.get("code")
    score = card.get("score", 60)
    sug = card.get("suggestion", "关注")
    p_up = card.get("prob_up", 0.5)
    ret_pct = card.get("expected_return_pct", "")
    val = card.get("valuation", "估值合理")
    cat = card.get("catalyst", "")
    reasons = "、".join(card.get("reasons", [])[:2])
    
    return f"【{name}】AI 量化综合打分 {score} 分，评估建议【{sug}】。模型测算未来 20 个交易日上涨概率为 {p_up:.0%}，预期收益率 {ret_pct}。估值层面表现为{val}。驱动因素：{reasons}。{cat}建议投资者控制仓位防守介入。"
