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
from ..model.lgbm import _feature_cols, quality_signal_rank

log = logging.getLogger(__name__)


def walk_forward_oos(train_days: int = 756, step: int = 21) -> pd.DataFrame:
    """滚动重训 + 样本外预测，返回拼接的 OOS 预测 DataFrame。"""
    import lightgbm as lgb

    cfg = get_settings()
    mat = read_parquet("features").set_index(["date", "code"]).sort_index()
    if mat.empty:
        raise RuntimeError("features 为空。")
    feats = _feature_cols(mat)
    params = dict(cfg.model.lightgbm)
    ranking = str(cfg.backtest.get("ranking", "label"))
    if ranking == "quality":
        pred = mat[["market"]].copy()
        pred["rank_quality"] = quality_signal_rank(mat)
        pred = pred.reset_index()
        pred["date"] = pred["date"].astype(str)
        pred["split"] = "test"
        test_start = pd.Timestamp(dict(cfg.model.split).get("test_start"))
        pred = pred[pd.to_datetime(pred["date"]) >= test_start]
        write_parquet(pred, "predictions_oos")
        return pred
    target = {"residual": "residual_return", "label": "label", "abs": "abs_label", "bench": "bench_label"}.get(ranking, "residual_return")

    dates = pd.to_datetime(mat.index.get_level_values("date")).unique().sort_values()
    seg = dict(cfg.model.split)
    test_start = pd.Timestamp(seg.get("test_start")) if seg.get("test_start") else dates[len(dates) // 3]
    test_dates = pd.to_datetime(pd.Series(dates))[pd.to_datetime(pd.Series(dates)) >= test_start].tolist()
    test_dates = sorted(set(test_dates))

    # embargo 隔离带（同主流程，系数 2.2 → 约 1.55× 交易日 horizon），扣除训练右端的未来标签泄漏；
    # 同时起到 purge 作用：训练集截止于 (win_end - emb)，确保其标签窗口不进入 OOS 区间。
    horizon = int(cfg.feature.get("label_horizon", 20))
    embargo_days = int(seg.get("embargo_days") or max(1, round(horizon * 2.2)))
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
        train_df = mat[train_idx].dropna(subset=[target])
        if train_df.empty or len(train_df) < 200:
            continue
        model_params = {k: v for k, v in params.items() if k != "objective"}
        if target == "residual_return":
            model = lgb.LGBMRegressor(**model_params, objective="regression_l1", verbose=-1)
            model.fit(train_df[feats], train_df[target].astype(float))
        else:
            tr = train_df.copy()
            tr["__date__"] = tr.index.get_level_values("date")
            tr = tr.sort_values(["__date__", "market"])
            groups = tr.groupby(["__date__", "market"], sort=False).size().to_numpy()
            model = lgb.LGBMRanker(**model_params, objective="lambdarank", verbose=-1)
            model.fit(tr[feats], tr[target].astype(int), group=groups)

        # 预测 [anchor, anchor+step) 的样本外区间
        oos_end = test_dates[min(i + step, len(test_dates)) - 1] + pd.Timedelta(days=1)
        oos_idx = (pd.to_datetime(mat.index.get_level_values("date")) >= win_end) & \
                  (pd.to_datetime(mat.index.get_level_values("date")) < oos_end)
        oos = mat[oos_idx]
        if oos.empty:
            continue
        raw = model.predict(oos[feats])
        rank = pd.Series(raw, index=oos.index).groupby(
            [oos.index.get_level_values("date"), oos["market"].fillna("other").to_numpy()], sort=False
        ).rank(pct=True).values
        p = oos[[target, "market"]].copy() if target in oos else pd.DataFrame(index=oos.index)
        p[f"rank_{target}"] = rank
        p = p.reset_index()
        p["date"] = p["date"].astype(str)
        preds.append(p)
        n_refit += 1

    if not preds:
        raise RuntimeError("walk-forward 未产出预测，检查 train_days/test_start。")
    pred = pd.concat(preds, ignore_index=True).dropna(subset=[f"rank_{target}"])
    pred["split"] = "test"
    write_parquet(pred, "predictions_oos")  # OOS 预测单独存(不覆盖 predictions，避免破坏日报 recs)
    log.info("[walkforward] 重训 %d 次, OOS 预测 %d 行", n_refit, len(pred))
    return pred


def run_walkforward(train_days: int = 756, step: int = 21) -> dict:
    """walk-forward OOS 预测 + 回测。"""
    walk_forward_oos(train_days=train_days, step=step)
    from .strategy import run_backtest

    report = run_backtest(pred_name="predictions_oos")
    report["mode"] = "walk_forward"
    report["train_days"] = train_days
    report["step"] = step
    # 落盘
    out = Path(get_settings().paths.output_dir) / "backtest_metrics.txt"
    import json
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report
