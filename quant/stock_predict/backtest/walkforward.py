"""Walk-forward 滚动回测（机构标准 OOS 评估）。

比单段切分更可信：每隔 step 个交易日，用过去 train_days 天重训模型，
预测随后 step 天（严格样本外），拼接成整段 OOS 预测，再回测。

杜绝「用未来数据训练」带来的乐观偏差。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get_settings
from ..data.warehouse import read_parquet, write_parquet

log = logging.getLogger(__name__)


def _feat_cols(df: pd.DataFrame) -> list[str]:
    meta = {"future_return", "industry_excess", "label", "industry", "market", "name", "market_cap"}
    return [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]


def walk_forward_oos(train_days: int = 756, step: int = 21) -> pd.DataFrame:
    """滚动重训 + 样本外预测，返回拼接的 OOS 预测 DataFrame。"""
    import lightgbm as lgb

    cfg = get_settings()
    mat = read_parquet("features").set_index(["date", "code"]).sort_index()
    if mat.empty:
        raise RuntimeError("features 为空。")
    feats = _feat_cols(mat)
    params = dict(cfg.model.lightgbm)

    dates = pd.to_datetime(mat.index.get_level_values("date")).unique().sort_values()
    seg = dict(cfg.model.split)
    test_start = pd.Timestamp(seg.get("test_start")) if seg.get("test_start") else dates[len(dates) // 3]
    test_dates = pd.to_datetime(pd.Series(dates))[pd.to_datetime(pd.Series(dates)) >= test_start].tolist()
    test_dates = sorted(set(test_dates))

    # 接入与主流程一致的 embargo 隔离区，扣除训练右端 28 天的未来标签泄漏
    horizon = int(cfg.feature.get("label_horizon", 20))
    embargo_days = int(seg.get("embargo_days") or max(1, round(horizon * 1.4)))
    emb = pd.Timedelta(days=embargo_days)

    preds = []
    n_refit = 0
    for i in range(0, len(test_dates), step):
        anchor = test_dates[i]
        win_end = anchor
        win_start = win_end - pd.Timedelta(days=int(train_days * 1.5))  # 日历日近似
        # 扣除 embargo 隔离带：训练集有效截止于 (win_end - emb)，彻底隔离标签穿越
        train_idx = (pd.to_datetime(mat.index.get_level_values("date")) >= win_start) & \
                    (pd.to_datetime(mat.index.get_level_values("date")) <= (win_end - emb))
        train_df = mat[train_idx].dropna(subset=["label"])
        if train_df.empty or len(train_df) < 200:
            continue
        model = lgb.LGBMClassifier(
            **{k: v for k, v in params.items() if k != "objective"}, objective="binary", verbose=-1
        )
        model.fit(train_df[feats], train_df["label"].astype(int))

        # 预测 [anchor, anchor+step) 的样本外区间
        oos_end = test_dates[min(i + step, len(test_dates)) - 1] + pd.Timedelta(days=1)
        oos_idx = (pd.to_datetime(mat.index.get_level_values("date")) >= win_end) & \
                  (pd.to_datetime(mat.index.get_level_values("date")) < oos_end)
        oos = mat[oos_idx]
        if oos.empty:
            continue
        proba = model.predict_proba(oos[feats])[:, 1]
        p = oos[["label"]].copy() if "label" in oos else pd.DataFrame(index=oos.index)
        p["prob"] = proba
        p = p.reset_index()
        p["date"] = p["date"].astype(str)
        preds.append(p)
        n_refit += 1

    if not preds:
        raise RuntimeError("walk-forward 未产出预测，检查 train_days/test_start。")
    pred = pd.concat(preds, ignore_index=True).dropna(subset=["prob"])
    pred["split"] = "test"
    write_parquet(pred, "predictions")  # 用 OOS 预测覆盖，供 backtest
    log.info("[walkforward] 重训 %d 次, OOS 预测 %d 行", n_refit, len(pred))
    return pred


def run_walkforward(train_days: int = 756, step: int = 21) -> dict:
    """walk-forward OOS 预测 + 回测。"""
    walk_forward_oos(train_days=train_days, step=step)
    from .strategy import run_backtest

    report = run_backtest()
    report["mode"] = "walk_forward"
    report["train_days"] = train_days
    report["step"] = step
    # 落盘
    out = Path(get_settings().paths.output_dir) / "backtest_metrics.txt"
    import json
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report
