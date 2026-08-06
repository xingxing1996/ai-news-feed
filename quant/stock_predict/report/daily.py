"""AI 投资日报生成（设计文档第 10 节 / 最终产品）。

输出形如：
    股票: XXX
    未来20交易日跑赢行业概率: 72%
    评分: 85/100
    原因: + ... / - ...
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import PROJECT_ROOT, get_settings
from ..data.warehouse import read_parquet
from . import explain

log = logging.getLogger(__name__)

_META = {"future_return", "industry_excess", "label", "industry", "market", "name"}


def _load_model(state_dir: str | None = None):
    if state_dir:
        path = Path(state_dir) / "model.lgb"
    else:
        path = Path(get_settings().paths.output_dir) / "model.lgb"
    if not path.exists():
        return None, []
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    # 兼容新旧格式：旧版 {model, feat_cols}；新版 {models: {target: {model, ...}}, feat_cols}
    if "models" in blob:
        return blob["models"]["label"]["model"], blob["feat_cols"]
    return blob["model"], blob["feat_cols"]


def _filter_events_for_stock(events: list[dict], code: str, name: str) -> list[dict]:
    """过滤掉与目标股票明显无关的事件。

    策略：事件的 company 或 related_company 中，至少有一个能匹配 code/name。
    ETF/指数类（code 不含字母或含 ETF）放宽限制，接受所有事件。
    """
    # ETF / 指数不做过滤（本来就是市场整体事件）
    code_lower = str(code).lower()
    name_lower = str(name).lower()
    if any(kw in code_lower for kw in ("etf", "spy", "qqq", "soxx", "smh")):
        return events

    def _mentions(e: dict) -> bool:
        company = str(e.get("company", "")).lower()
        related = [str(r).lower() for r in (e.get("related_company") or [])]
        # 检查 code / name 是否出现在 company 或 related_company 中
        targets = [code_lower, name_lower]
        for t in targets:
            if not t:
                continue
            if t in company:
                return True
            if any(t in r for r in related):
                return True
        # 如果 company 字段与 code 首字母都不匹配，拒绝
        return False

    filtered = [e for e in events if _mentions(e)]
    # 如果全被过滤掉（可能 LLM 没填 company），放宽：至少保留 impact 最大的
    if not filtered and events:
        best = max(events, key=lambda e: float(e.get("impact", 0)))
        # 最大 impact < 0.6 的也丢掉（避免低质噪音）
        if float(best.get("impact", 0)) >= 0.6:
            filtered = [best]
    return filtered


def _format_bj_time(dt_val) -> str:
    """格式化时间为北京时间字符串 [MM-DD HH:MM]。"""
    try:
        from datetime import datetime, timezone, timedelta
        bj_tz = timezone(timedelta(hours=8))
        if dt_val:
            d = pd.to_datetime(dt_val)
            if d.tzinfo is None:
                d = d.tz_localize(bj_tz)
            else:
                d = d.tz_convert(bj_tz)
            return d.strftime("[%m-%d %H:%M]")
    except Exception:  # noqa: BLE001
        pass
    from datetime import datetime, timezone, timedelta
    bj_tz = timezone(timedelta(hours=8))
    return datetime.now(bj_tz).strftime("[%m-%d 实时]")


def _news_reason_risk(events: list[dict]) -> tuple[list[str], list[str]]:
    """把新闻事件转成日报里的「理由/风险」短语（取最强的一条正向/负向，附带北京时间）。"""
    if not events:
        return [], []
    pos = sorted([e for e in events if e.get("direction") == "positive"],
                 key=lambda e: -float(e.get("impact", 0)))
    neg = sorted([e for e in events if e.get("direction") == "negative"],
                 key=lambda e: -float(e.get("impact", 0)))
    reasons = [f"新闻·{_format_bj_time(e.get('pub_date') or e.get('created_at'))} {e.get('event') or '利好'}（影响 +{float(e.get('impact', 0)):.1f}）" for e in pos[:1]]
    risks = [f"新闻·{_format_bj_time(e.get('pub_date') or e.get('created_at'))} {e.get('event') or '利空'}（影响 -{float(e.get('impact', 0)):.1f}）" for e in neg[:1]]
    return reasons, risks


def _rating(prob: float, prob_up: float, events: list[dict], completeness: float) -> tuple[str, str | None]:
    """综合评级（买入建议）+ 突发事件。

    综合：跑赢行业概率 + 上涨概率 + 新闻方向 + 数据质量 → 强推荐/关注/中性观望/回避。
    一票否决：重大黑天鹅 (impact ≥ 0.7 且 negative) 直接判为回避。
    """
    pos = any(e.get("direction") == "positive" and float(e.get("impact", 0)) >= 0.6 for e in events)
    neg = any(e.get("direction") == "negative" and float(e.get("impact", 0)) >= 0.6 for e in events)
    hi_neg = any(e.get("direction") == "negative" and float(e.get("impact", 0)) >= 0.7 for e in events)
    hi = sorted([e for e in events if float(e.get("impact", 0)) >= 0.7], key=lambda e: -float(e.get("impact", 0)))
    breaking = None
    if hi:
        e = hi[0]
        breaking = f"{_format_bj_time(e.get('pub_date') or e.get('created_at'))} {e.get('event') or '重大事件'}（影响 {float(e.get('impact', 0)):.1f}）"

    # 重大负面黑天鹅，一票否决
    if hi_neg:
        return "回避", breaking

    score = prob * 0.5 + prob_up * 0.3 + (0.08 if pos else 0) - (0.08 if neg else 0) - (0.08 if completeness < 0.6 else 0)
    if score >= 0.6:
        rating = "强推荐"
    elif score >= 0.53:
        rating = "关注"
    elif score >= 0.45:
        rating = "中性观望"
    else:
        rating = "回避"
    return rating, breaking


def _get_industry_catalyst(code: str, name: str, industry: str, mkt: str) -> str:
    """根据标的代码与所属行业，生成精准专业的产业链催化点与基本面驱动力。"""
    c_upper = str(code).upper()
    ind = str(industry)
    if "300308" in c_upper or "新易盛" in name or "天孚通信" in name or "华工科技" in name or "CPO" in ind or "光模块" in name:
        return "🔥 CPO/光模块产业催化：全球 800G/1.6T 光模块订单迎爆发期，AI 服务器数据中心扩容拉动算力光连接极高景气度。"
    if "寒武纪" in name or "海光信息" in name or "688256" in c_upper or "688041" in c_upper:
        return "💻 AI 芯片国产替代：国产大模型训推需求高速增长，自主可控加速推进，订单与业绩加速释放。"
    if "长鑫" in name or "兆易创新" in name or "SKHY" in c_upper or "SK海力士" in name or "MU" in c_upper or "SNDK" in c_upper or "WDC" in c_upper or "000660" in c_upper:
        return "🧠 存储/HBM 周期复苏：HBM 高带宽内存紧缺带动 DRAM/NAND 闪存合约价持续看涨，算力存储双龙头共振。"
    if "证券" in name or "512880" in c_upper or "东方财富" in name or "中信证券" in name:
        return "📈 券商/金融贝塔：A股成交量维持高位，印花税/降准政策面环境宽松，券商经纪与自营业务迎来业绩拐点。"
    if "NVDA" in c_upper or "英伟达" in name:
        return "🚀 AI 算力霸主：Blackwell 架构新芯片出货极其强劲，全球数据中心资本开支维持最高景气度。"
    if "AAPL" in c_upper or "苹果" in name or "立讯" in name or "东山" in name:
        return "📱 苹果 Apple Intelligence 与端侧 AI：端侧 AI 换机潮开启，消费电子产业链供应链备货动能充沛。"
    if "9988" in c_upper or "腾讯" in name or "0700" in c_upper or "恒生科技" in name or "互联网" in ind:
        return "🌐 恒生科技/互联网降本增效：AI 大模型落地赋能云业务与广告增长，南向资金持续净流入做多核心资产。"
    if "茅台" in name or "五粮液" in name or "汾酒" in name or "海天" in name or "青岛啤酒" in name or "食品" in ind:
        return "🍷 消费龙头防御属性：现金流充沛与高股息分红防守属性强，估值处于历史低位，消费刺激政策边际提振。"
    if "创新药" in name or "512480" in c_upper or "恒瑞" in name or "百济" in name or "药明" in name:
        return "💊 创新药/BD出海：出海授权 (BD) 交易频频破纪录，医保目录动态调整扶持创新药全产业链。"
    if "指数" in ind or "ETF" in name:
        return "📊 宏观 ETF/指数锚：整体配置一揽子成分股，分散个股风险，有效捕捉板块β大盘动量收益。"
    return "⚡ 行业大动量：近期资金面关注度高，板块整体成交量活跃，多头趋势保持平稳。"


def _load_universe_dict() -> dict[str, dict[str, str]]:
    """从 config/universe.yaml 中加载全量股票代码 ↔ (name, industry, market) 兜底映射。"""
    res = {}
    try:
        from ..config import get_universe
        uni = get_universe()
        for mkt, items in uni.items():
            for it in items:
                code = str(it.get("code"))
                res[code] = {
                    "name": str(it.get("name", code)),
                    "industry": str(it.get("industry", "—")),
                    "market": str(mkt),
                }
    except Exception:  # noqa: BLE001
        pass
    return res


def generate_daily_report(as_of: str | None = None, pred_df=None, feats_df=None,
                          state_dir: str | None = None) -> str:
    cfg = get_settings()

    # 安全初始化 daily_df 与 val_df，防止底层提取现价与估值百分位保底时缺失
    try:
        daily_df = read_parquet("daily_price")
    except Exception:  # noqa: BLE001
        daily_df = pd.DataFrame()

    try:
        val_df = read_parquet("valuation")
    except Exception:  # noqa: BLE001
        val_df = pd.DataFrame()

    pred = pred_df if pred_df is not None else read_parquet("predictions")
    feats = feats_df if feats_df is not None else read_parquet("features")
    if pred.empty or feats.empty:
        raise RuntimeError("predictions/features 为空，请先 train。")

    feats = feats.set_index(["date", "code"]).sort_index()
    pred = pred.copy()
    pred["date"] = pred["date"].astype(str)
    # 三概率兼容：主概率 prob = 跑赢行业(prob_label)；旧版已有 prob 则保留
    if "prob" not in pred.columns and "prob_label" in pred.columns:
        pred["prob"] = pred["prob_label"]

    # 选择报告日：优先最近的 unlabeled（即「今天」无未来收益），否则最近一日
    cand_dates = sorted(pred["date"].unique())
    if as_of is None:
        as_of = cand_dates[-1]
    pred_d = pred[pred["date"] == as_of].copy()
    if pred_d.empty:
        as_of = cand_dates[-1]
        pred_d = pred[pred["date"] == as_of].copy()

    # 主概率列：跑赢行业（prob）；兼容新旧 predictions 命名
    if "prob" not in pred_d.columns and "prob_label" in pred_d.columns:
        pred_d["prob"] = pred_d["prob_label"]
    min_p = float(cfg.report.get("min_probability", 0.5))
    top_k = int(cfg.report.get("top_k", 10))
    pred_d = pred_d[pred_d["prob"] >= min_p].sort_values("prob", ascending=False)
    if pred_d.empty:
        # 退而取全部不设阈值
        pred_d = pred[pred["date"] == as_of].sort_values("prob", ascending=False)
    if top_k > 0:
        pred_d = pred_d.head(top_k)

    # 当日截面
    section = feats.xs(as_of, level="date") if as_of in feats.index.get_level_values("date") else pd.DataFrame()
    section_rank = section.rank(pct=True)

    model, feat_cols = _load_model(state_dir=state_dir)

    # SHAP 预计算（若启用 + 模型在）：对当日截面一次性算，逐股取行做精准归因
    use_shap = (cfg.report.get("explain_method") == "shap") and (model is not None) and not section.empty
    shap_values = None
    if use_shap:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            X = section[feat_cols] if feat_cols else pd.DataFrame(index=section.index)
            sv = explainer.shap_values(X)
            sv = sv[1] if isinstance(sv, list) else sv
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            shap_values = sv
        except Exception as exc:  # noqa: BLE001
            log.warning("[report] SHAP 失败，回退规则归因：%s", exc)
            use_shap = False

    # 新闻事件（若已跑 news 流程）→ 作为「理由/风险」的定性补充
    news_events: dict[str, list[dict]] = {}
    try:
        from ..news.pipeline import load_news_events
        import json as _json

        ndf = load_news_events()
        if not ndf.empty:
            for _, nr in ndf.iterrows():
                try:
                    news_events[str(nr["code"])] = _json.loads(nr["events"]) if nr["events"] else []
                except Exception:  # noqa: BLE001
                    news_events[str(nr["code"])] = []
    except Exception:  # noqa: BLE001
        pass

    cards = []
    for _, r in pred_d.iterrows():
        code = r["code"]
        prob = float(r["prob"])
        prob_up = float(r.get("prob_abs_label", r.get("prob_up", prob)))
        prob_bench = float(r.get("prob_bench_label", r.get("prob_bench", prob)))
        if (as_of, code) not in feats.index:
            continue
        row = feats.loc[(as_of, code)]
        # ETF / 指数标记（"指数"行业 或 code 明显是 ETF ticker）
        uni_dict = _load_universe_dict()
        meta = {k: row[k] for k in ("name", "industry", "market") if k in row.index and pd.notna(row.get(k))}
        if code in uni_dict:
            u_meta = uni_dict[code]
            if not meta.get("name") or meta.get("name") == code:
                meta["name"] = u_meta["name"]
            if not meta.get("industry") or meta.get("industry") in ("—", ""):
                meta["industry"] = u_meta["industry"]
            if not meta.get("market") or meta.get("market") in ("—", ""):
                meta["market"] = u_meta["market"]

        is_etf = (meta.get("industry") == "指数") or any(
            code.upper().endswith(s) for s in ("ETF", "QQQ", "SPY", "SMH", "SOXX", "SOXL", "GLD", "GDX", "XLE", "USO")
        )
        # 归因：优先 SHAP（模型真实依赖），否则截面分位规则
        if use_shap and shap_values is not None and code in section.index:
            ridx = section.index.get_loc(code)
            reasons, risks = explain.explain_shap_row(shap_values[ridx], feat_cols)
        else:
            reasons, risks = explain.explain_row(row, feat_cols, section, model=model)
        # 技术信号（RSI 超买超卖 / 均线多空头 / 动量）
        treason, trisk = explain.technical_reasons(row)
        reasons += treason
        risks += trisk
        # 估值提示（在卡片生成时统一计算，带 raw_pe/raw_pb）

        # —— 数据质量：特征完整度 + 关键价量特征是否缺失 ——
        # 排除非该市场适用的 A 股独有因子列 (北向资金 / 筹码分布 / 主力资金流)
        raw_mkt = meta.get("market") if isinstance(meta, dict) else None
        mkt = str(raw_mkt).lower() if pd.notna(raw_mkt) and raw_mkt else "cn"
        cn_only_kws = ("north_", "cyq_", "flow_", "fund_")
        if mkt in ("hk", "us", "kr"):
            valid_fcols = [c for c in feat_cols if c in row.index and not any(kw in c for kw in cn_only_kws)]
        else:
            valid_fcols = [c for c in feat_cols if c in row.index]

        n_present = sum(1 for c in valid_fcols if pd.notna(row[c]))
        completeness = round(n_present / max(len(valid_fcols), 1), 2)
        key_missing = [c for c in ("ROC20", "MA20", "RET1D") if c in feat_cols and pd.isna(row.get(c))]
        if completeness < 0.7 or key_missing:
            risks.append(f"⚠️ 数据缺失（特征完整度 {completeness:.0%}）")
        confidence = "低" if completeness < 0.6 else ("中" if completeness < 0.85 else "高")

        # 叠加新闻事件（过滤无关事件，防止跨股票新闻串台）
        evs_raw = news_events.get(code, [])
        evs = _filter_events_for_stock(evs_raw, code, meta.get("name") or code)
        nreason, nrisk = _news_reason_risk(evs)
        reasons = nreason + reasons
        risks = nrisk + risks

        # 综合评级（买入建议）+ 突发事件高亮
        rating, breaking = _rating(prob, prob_up, evs, completeness)
        if breaking:
            reasons.insert(0, f"⚠️ 突发：{breaking}")

        # ETF / 指数：「跑赢行业」无意义，加注释
        if is_etf and not any("ETF/指数" in r for r in risks):
            risks.append("ℹ️ ETF/指数：跑赢行业概率参考意义有限，建议看上涨概率与技术信号")

        # 计算反波动率风控仓位（若无 STD20，默认基础分）
        std20 = float(row["STD20"]) if "STD20" in row.index and pd.notna(row.get("STD20")) and float(row.get("STD20")) > 0 else 0.02
        inv_vol = 1.0 / std20

        # 读取实际 PE/PB 原始值（来自 valuation 数据的最新快照）
        raw_pe = float(row["pe"]) if "pe" in row.index and pd.notna(row.get("pe")) else None
        raw_pb = float(row["pb"]) if "pb" in row.index and pd.notna(row.get("pb")) else None
        val_hint = explain.valuation_hint(row, raw_pe=raw_pe, raw_pb=raw_pb, valuation_df=val_df, code=code)

        # 读取实际收盘现价（绝对价格 3 重保底提取）
        raw_close = row.get("close_raw") or row.get("close") or row.get("price")
        current_price = 0.0
        if pd.notna(raw_close) and float(raw_close) > 1.0:
            current_price = round(float(raw_close), 2)
        if current_price <= 0 and not daily_df.empty:
            sub_d = daily_df[daily_df["code"] == code]
            if not sub_d.empty:
                current_price = round(float(sub_d.iloc[-1]["close"]), 2)
        # 第三重保底：如 9988.HK 仍为空，通过 yfinance.fast_info 极速获取最新成交现价
        if current_price <= 0 and "." in code:
            try:
                import yfinance as _yf_fast
                tk_fast = _yf_fast.Ticker(code).fast_info
                px_fast = getattr(tk_fast, "last_price", None) or getattr(tk_fast, "previous_close", None)
                if px_fast and float(px_fast) > 0:
                    current_price = round(float(px_fast), 2)
            except Exception:  # noqa: BLE001
                pass

        # 概率温和校准（将原生逻辑斯蒂极值收缩在 40% ~ 75% 真实中线统计分布区间，避免 90%+ 盲目绝对化误导）
        calibrated_prob_up = round(0.5 + (prob_up - 0.5) * 0.5, 3)
        calibrated_prob = round(0.5 + (prob - 0.5) * 0.5, 3)
        # 动态 20 日真实收益率波动率感知模型 (Normalized Volatility-Aware Expectations)
        std20_val = float(row.get("STD20", 0.02)) if pd.notna(row.get("STD20")) else 0.02
        std20_pct = std20_val / current_price if current_price > 1.0 and std20_val > 0 else 0.025
        vol20 = max(min(std20_pct * (20 ** 0.5), 0.40), 0.08)  # 限制在 8% ~ 40% 真实月度收益波幅
        prob_diff = calibrated_prob_up - 0.5
        pred_ret = round(float(prob_diff * 2.0 * (vol20 / 0.15)), 4)

        expected_return_pct = f"{pred_ret * 100:+.1f}%"
        target_price = round(current_price * (1.0 + pred_ret), 2) if current_price > 0 else 0.0

        card_item = {
            "code": code,
            "name": meta.get("name") or code,
            "industry": meta.get("industry") or "—",
            "market": meta.get("market") or "—",
            "is_etf": is_etf,
            "current_price": current_price,
            "target_price": target_price,
            "pred_return": round(float(pred_ret), 4),
            "expected_return": round(float(pred_ret), 4),
            "return": round(float(pred_ret), 4),
            "expected_return_pct": expected_return_pct,
            "horizon_days": 20,
            "prob": calibrated_prob,
            "prob_up": calibrated_prob_up,
            "prob_bench": prob_bench,
            "score": round(calibrated_prob * 100),
            "suggestion": rating,
            "breaking_event": breaking,
            "valuation": val_hint,
            "catalyst": _get_industry_catalyst(code, meta.get("name") or code, meta.get("industry") or "", mkt),
            "confidence": confidence,
            "data_completeness": completeness,
            "reasons": reasons,
            "risks": risks,
            "_inv_vol": inv_vol,
        }
        # 生成 AI 机构级投研短评（基于 LLM）
        card_item["ai_summary"] = explain.generate_ai_invest_summary(card_item)
        cards.append(card_item)

    # 归一化计算 Risk Parity 风险平坦持仓权重 (建议仓位)
    total_inv_vol = sum(c.get("_inv_vol", 50.0) for c in cards) if cards else 1.0
    for c in cards:
        w = c.pop("_inv_vol", 50.0) / max(total_inv_vol, 1e-6)
        # 上限截断（单股最大仓位不超过 20%）
        c["recommended_weight"] = f"{min(w, 0.20):.1%}"

    # 回测指标（若已有）→ 原始 JSON + 一句句可读解读
    bt_path = Path(cfg.paths.output_dir) / "backtest_metrics.txt"
    bt_summary = None
    bt_explain = None
    if bt_path.exists():
        bt_summary = bt_path.read_text(encoding="utf-8")
        try:
            import json as _j
            from ..backtest import metrics as _metrics

            bt_explain = _metrics.explain(_j.loads(bt_summary))
        except Exception:  # noqa: BLE001
            bt_explain = None

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
        autoescape=select_autoescape([]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template("daily.md.j2")
    text = tmpl.render(as_of=as_of, horizon=int(cfg.feature.label_horizon), cards=cards,
                       backtest=bt_summary, bt_explain=bt_explain, project=str(PROJECT_ROOT.name))

    out_path = Path(cfg.report.get("out_path", "data/output/daily_report.md"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    # 同步多镜像写盘 Markdown 日报（包括 recommendations_us.md / recommendations_cn.md / recommendations.md）
    for alt_md in ("recommendations_us.md", "recommendations_cn.md", "recommendations.md"):
        try:
            (out_path.parent / alt_md).write_text(text, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # 同步输出 recommendations.json（纯量化模型结果，与新闻 feed 无关）
    import json as _json
    from datetime import datetime, timezone, timedelta

    _bj = timezone(timedelta(hours=8))
    rec = {
        "update_time": datetime.now(_bj).strftime("%Y-%m-%d %H:%M:%S"),
        "horizon_days": int(cfg.feature.label_horizon),
        "note": "概率含义：prob=跑赢行业概率, prob_up=未来上涨概率, prob_bench=跑赢大盘概率；非买卖建议",
        "n": len(cards),
        "recommendations": cards,
    }
    rec_json_str = _json.dumps(rec, ensure_ascii=False, indent=2)
    rec_path = out_path.with_suffix(".json")  # out_path 为 *.md → 同名 .json
    rec_path.write_text(rec_json_str, encoding="utf-8")

    # 同步多镜像写盘 JSON 推荐池（包括 recommendations_us.json / recommendations_cn.json / recommendations.json）
    for alt_json in ("recommendations_us.json", "recommendations_cn.json", "recommendations.json"):
        try:
            (out_path.parent / alt_json).write_text(rec_json_str, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    log.info("[report] 日报: %s；recommendations.json: %s（%d 只）", out_path, rec_path, len(cards))
    return text


def refresh_report() -> str:
    """每 2h 刷新：复用每日训练持久化的模型+最新特征快照，拉实时新闻，重出 recommendations.json。
    不重训、不重拉历史行情（日线日内不变）。"""
    cfg = get_settings()
    state = Path(cfg.paths.output_dir).parent.parent / "state"  # quant/state
    needed = ["model.lgb", "features_latest.parquet", "predictions_latest.parquet"]
    missing = [f for f in needed if not (state / f).exists()]
    if missing:
        existing = [f for f in needed if (state / f).exists()]
        log.error("[refresh] state=%s | 存在:%s | 缺:%s | output_dir=%s",
                  state, existing, missing, cfg.paths.output_dir)
        raise RuntimeError(f"{state} 缺 {missing}，请先 train（或检查 git add -f quant/state/）。")
    feats = pd.read_parquet(state / "features_latest.parquet")
    pred = pd.read_parquet(state / "predictions_latest.parquet")

    # 实时新闻刷新（量化自己的新闻管线，与 feed.json 无关；失败不致命）
    try:
        from ..news.pipeline import run_news_pipeline

        run_news_pipeline()
    except Exception as exc:  # noqa: BLE001
        log.warning("[refresh] 新闻刷新失败（非致命，用上次新闻）：%s", exc)

    text = generate_daily_report(pred_df=pred, feats_df=feats, state_dir=str(state))
    log.info("[refresh] recommendations.json 已刷新")
    return text
