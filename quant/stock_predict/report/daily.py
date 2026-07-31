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
    return blob["model"], blob["feat_cols"]


def _news_reason_risk(events: list[dict]) -> tuple[list[str], list[str]]:
    """把新闻事件转成日报里的「理由/风险」短语（取最强的一条正向/负向）。"""
    if not events:
        return [], []
    pos = sorted([e for e in events if e.get("direction") == "positive"],
                 key=lambda e: -float(e.get("impact", 0)))
    neg = sorted([e for e in events if e.get("direction") == "negative"],
                 key=lambda e: -float(e.get("impact", 0)))
    reasons = [f"新闻·{e.get('event') or '利好'}（影响 {float(e.get('impact', 0)):.1f}）" for e in pos[:1]]
    risks = [f"新闻·{e.get('event') or '利空'}（影响 {float(e.get('impact', 0)):.1f}）" for e in neg[:1]]
    return reasons, risks


def generate_daily_report(as_of: str | None = None, pred_df=None, feats_df=None,
                          state_dir: str | None = None) -> str:
    cfg = get_settings()
    pred = pred_df if pred_df is not None else read_parquet("predictions")
    feats = feats_df if feats_df is not None else read_parquet("features")
    if pred.empty or feats.empty:
        raise RuntimeError("predictions/features 为空，请先 train。")

    feats = feats.set_index(["date", "code"]).sort_index()
    pred["date"] = pred["date"].astype(str)

    # 选择报告日：优先最近的 unlabeled（即「今天」无未来收益），否则最近一日
    cand_dates = sorted(pred["date"].unique())
    if as_of is None:
        as_of = cand_dates[-1]
    pred_d = pred[pred["date"] == as_of].copy()
    if pred_d.empty:
        as_of = cand_dates[-1]
        pred_d = pred[pred["date"] == as_of].copy()

    min_p = float(cfg.report.get("min_probability", 0.5))
    top_k = int(cfg.report.get("top_k", 10))
    pred_d = pred_d[pred_d["prob"] >= min_p].sort_values("prob", ascending=False).head(top_k)
    if pred_d.empty:
        # 退而取 Top-K 不设阈值
        pred_d = pred[pred["date"] == as_of].sort_values("prob", ascending=False).head(top_k)

    # 当日截面
    section = feats.xs(as_of, level="date") if as_of in feats.index.get_level_values("date") else pd.DataFrame()
    section_rank = section.rank(pct=True)

    model, feat_cols = _load_model(state_dir=state_dir)

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
        if (as_of, code) not in feats.index:
            continue
        row = feats.loc[(as_of, code)]
        reasons, risks = explain.explain_row(row, feat_cols, section, model=model)

        # —— 数据质量：特征完整度 + 关键价量特征是否缺失（缺数据→标低可信）——
        fcols = [c for c in feat_cols if c in row.index]
        n_present = sum(1 for c in fcols if pd.notna(row[c]))
        completeness = round(n_present / max(len(feat_cols), 1), 2)
        key_missing = [c for c in ("ROC20", "MA20", "RET1D") if c in feat_cols and pd.isna(row.get(c))]
        if completeness < 0.7 or key_missing:
            risks.append(f"⚠️ 数据缺失（特征完整度 {completeness:.0%}）")
        confidence = "低" if completeness < 0.6 else ("中" if completeness < 0.85 else "高")

        # 叠加新闻事件（设计文档：新闻应给「HBM需求增加」这类上下文理由）
        nreason, nrisk = _news_reason_risk(news_events.get(code, []))
        reasons = nreason + reasons
        risks = nrisk + risks

        meta = {k: row[k] for k in ("name", "industry", "market") if k in row.index}
        cards.append(
            {
                "code": code,
                "name": meta.get("name") or code,
                "industry": meta.get("industry") or "—",
                "market": meta.get("market") or "—",
                "prob": prob,
                "score": round(prob * 100),
                "confidence": confidence,
                "data_completeness": completeness,
                "reasons": reasons,
                "risks": risks,
            }
        )

    # 回测指标（若已有）
    bt_path = Path(cfg.paths.output_dir) / "backtest_metrics.txt"
    bt_summary = bt_path.read_text(encoding="utf-8") if bt_path.exists() else None

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
        autoescape=select_autoescape([]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template("daily.md.j2")
    text = tmpl.render(as_of=as_of, horizon=int(cfg.feature.label_horizon), cards=cards,
                       backtest=bt_summary, project=str(PROJECT_ROOT.name))

    out_path = Path(cfg.report.get("out_path", "data/output/daily_report.md"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    # 同步输出 recommendations.json（纯量化模型结果，与新闻 feed 无关）
    import json as _json
    from datetime import datetime, timezone, timedelta

    _bj = timezone(timedelta(hours=8))
    rec = {
        "update_time": datetime.now(_bj).strftime("%Y-%m-%d %H:%M:%S"),
        "horizon_days": int(cfg.feature.label_horizon),
        "note": "量化模型(LightGBM+XGBoost)预测：未来 horizon 日跑赢行业的概率；非买卖建议",
        "n": len(cards),
        "recommendations": cards,
    }
    rec_path = out_path.parent / "recommendations.json"
    rec_path.write_text(_json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("[report] 日报: %s；recommendations.json: %s（%d 只）", out_path, rec_path, len(cards))
    return text


def refresh_report() -> str:
    """每 2h 刷新：复用每日训练持久化的模型+最新特征快照，拉实时新闻，重出 recommendations.json。
    不重训、不重拉历史行情（日线日内不变）。"""
    cfg = get_settings()
    state = Path(cfg.paths.output_dir).parent.parent / "state"  # quant/state
    if not (state / "model.lgb").exists() or not (state / "features_latest.parquet").exists():
        raise RuntimeError(f"{state} 下缺 model.lgb/features_latest.parquet，请先跑每日训练(train)。")
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
