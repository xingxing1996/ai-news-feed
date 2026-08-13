"""LightGBM 训练 / 预测。

输出「未来 horizon 日跑赢行业的横截面排名分位」。
- 按日期段切分 train / valid / test（避免未来穿越）。
- LightGBM 原生处理 NaN，故保留缺失值。
- 预测概率落到 ``warehouse/predictions.parquet``，模型存 ``output_dir/model.lgb``。
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get_settings
from ..data.warehouse import read_parquet, write_parquet
from . import evaluate

log = logging.getLogger(__name__)

_META_COLS = {"future_return", "industry_excess", "industry_excess_neu", "residual_return", "label",
              "abs_label", "bench_label", "bench_excess", "bench_future",
              "industry", "market", "name", "market_cap",
              "pe", "pb"}  # pe/pb 原值只供日报展示，不进模型特征


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in _META_COLS and not c.endswith("_raw") and pd.api.types.is_numeric_dtype(df[c])
    ]


def _segment_mask(dates: pd.Series, seg: dict, key: str) -> pd.Series:
    start = seg.get(f"{key}_start")
    end = seg.get(f"{key}_end")
    m = pd.Series(True, index=dates.index)
    if start:
        m &= dates >= pd.Timestamp(start)
    if end:
        m &= dates <= pd.Timestamp(end)
    return m


def _split(df: pd.DataFrame, seg: dict, embargo_days: int = 0) -> dict[str, pd.DataFrame]:
    """按日期段切分。embargo_days：段间隔离（日历日），消除重叠加标签的未来穿越。

    label horizon=20 会让 train 末尾样本的 label 用到 valid 期的收益 → 泄漏。
    加 embargo：train/valid 的有效截止日各前移 embargo，留出隔离带。
    """
    dates = pd.Series(pd.to_datetime(df.index.get_level_values("date")), index=df.index)
    emb = pd.Timedelta(days=embargo_days) if embargo_days else pd.Timedelta(0)
    out = {}
    out["train"] = df[dates <= (pd.Timestamp(seg["train_end"]) - emb)]
    v_mask = (_segment_mask(dates, seg, "valid")
              & (dates <= (pd.Timestamp(seg["valid_end"]) - emb)))
    out["valid"] = df[v_mask]
    if seg.get("test_end"):
        out["test"] = df[_segment_mask(dates, seg, "test")]
    elif seg.get("test_start"):
        out["test"] = df[dates >= pd.Timestamp(seg["test_start"])]
    else:
        out["test"] = df.iloc[0:0]

    # 智能自适应保底：若硬编码日期切出空训练集（如样本年份靠后），按 60/20/20 动态切分，
    # 同样施加 embargo 隔离带（与主路径一致，杜绝标签穿越）；样本不足以切分则报错而非全集合泄漏。
    if out["train"].empty:
        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)
        if n_dates < 3:
            raise RuntimeError(f"样本日期数({n_dates})不足以做 train/valid/test 切分，请检查数据范围")
        idx1 = int(n_dates * 0.6)
        idx2 = int(n_dates * 0.8)
        d1, d2 = unique_dates[idx1], unique_dates[idx2]
        out["train"] = df[dates < (d1 - emb)]
        out["valid"] = df[(dates >= d1) & (dates < (d2 - emb))]
        out["test"] = df[dates >= d2]
    return out


_TARGETS = ["residual_return"]  # 连续市场中性超额收益：训练、排序、回测共用同一目标
_QUALITY_FEATURES = ("gross_margin", "roe", "revenue_growth", "quality_rank")


def active_targets(cfg) -> list[str]:
    """Select the model target from the market-specific ranking configuration."""
    mode = str(cfg.backtest.get("ranking", "label"))
    if mode in ("residual", "quality"):
        return ["residual_return"]
    return [{"label": "label", "abs": "abs_label", "bench": "bench_label"}.get(mode, "label")]


def quality_signal_rank(df: pd.DataFrame) -> pd.Series:
    """PIT quality composite used as the deployable primary signal.

    Each component is ranked only within its date/market cross-section, then
    equally combined and ranked again. No fitted parameter or future return is
    involved, so the same computation is valid in every walk-forward window.
    """
    cols = [c for c in _QUALITY_FEATURES if c in df.columns]
    if not cols:
        raise RuntimeError("质量信号缺少 gross_margin/roe/revenue_growth/quality_rank 特征")
    markets = df["market"].fillna("other").to_numpy()
    dates = df.index.get_level_values("date")
    component_rank = df[cols].groupby([dates, markets], sort=False).rank(pct=True)
    score = component_rank.mean(axis=1)
    return score.groupby([dates, markets], sort=False).rank(pct=True)


def _train_one(feat_cols, params, splits, target, use_ensemble=True, calibrate=False):
    """为单个 target 训练 LightGBM Ranker，输出按日截面归一化的排名分位。

    ranker 输出 raw score → 按 date 截面 rank(pct=True) ∈(0,1]。这是排序信号，
    不是发生概率。target 的 0/1(zone 标签)
    作 lambdarank 的 relevance，每个交易日为一个 query group。
    """
    import lightgbm as lgb  # 延迟导入

    def _xy(df: pd.DataFrame):
        df = df.dropna(subset=[target]).copy()
        if target == "residual_return":
            return df[feat_cols], df[target].astype(float), None
        df["__date__"] = df.index.get_level_values("date")
        df = df.sort_values(["__date__", "market"])
        groups = df.groupby(["__date__", "market"], sort=False).size().to_numpy()
        return df[feat_cols], df[target].astype(int), groups

    Xtr, ytr, group_tr = _xy(splits["train"])
    Xva, yva, group_va = _xy(splits["valid"])
    if Xtr.empty:
        raise RuntimeError(f"训练集为空（{target}，train_end={splits['_seg'].get('train_end')}）")

    model_params = {k: v for k, v in params.items() if k != "objective"}
    is_continuous = target == "residual_return"
    model = (lgb.LGBMRegressor(**model_params, objective="regression_l1", verbose=-1)
             if is_continuous else lgb.LGBMRanker(**model_params, objective="lambdarank", verbose=-1))
    fit_kwargs: dict = {} if is_continuous else {"group": group_tr}
    if not Xva.empty:
        fit_kwargs.update({"eval_set": [(Xva, yva)], "callbacks": [lgb.early_stopping(50, verbose=False)]})
        if not is_continuous:
            fit_kwargs["eval_group"] = [group_va]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr, ytr, **fit_kwargs)

    def _rank(df_sub: pd.DataFrame):
        X = df_sub[feat_cols]
        raw = model.predict(X)
        # 截面 rank 归一化：只表达同日股票池内的相对排序。
        dates = X.index.get_level_values("date")
        markets = df_sub["market"].fillna("other").to_numpy()
        return pd.Series(raw, index=X.index).groupby([dates, markets], sort=False).rank(pct=True).values

    rank = _rank(splits["_all"])
    blob = {"model": model, "feat_cols": list(feat_cols),
            "kind": "residual_regressor" if is_continuous else "ranker",
            "used_ensemble": False, "calibrated": False}
    return model, rank, blob, len(Xtr)


def train_and_predict() -> dict:
    cfg = get_settings()
    targets = active_targets(cfg)
    mat = read_parquet("features")
    if mat.empty:
        raise RuntimeError("features 为空，请先 `stock-predict features`。")
    mat = mat.set_index(["date", "code"]).sort_index()
    # 防御：清洗 inf→NaN（兜底，即使上游 features 已清洗），防止 XGBoost 报错 / LightGBM 受 inf 污染
    _num = mat.select_dtypes(include="number").columns
    if len(_num):
        mat[_num] = mat[_num].replace([np.inf, -np.inf], np.nan)

    feat_cols = _feature_cols(mat)
    log.info("[model] 特征数=%d", len(feat_cols))

    seg = dict(cfg.model.split)
    # embargo：按 label horizon 隔离段间重叠（防标签穿越泄漏）。
    # 注意 horizon 是交易日(≈1.4日历日/天)，embargo_days 是日历日；系数 2.2 → 约 1.55× 交易日 horizon，
    # 留足缓冲（机构标准要求 embargo > label 窗口，López de Prado）。
    horizon = int(cfg.feature.get("label_horizon", 20))
    embargo_days = int(seg.get("embargo_days") or max(1, round(horizon * 2.2)))
    splits = _split(mat, seg, embargo_days=embargo_days)
    splits["_seg"] = seg
    splits["_all"] = mat
    params = dict(cfg.model.lightgbm)
    use_ensemble = bool(cfg.model.get("ensemble", True))
    calibrate = bool(cfg.model.get("calibrate", False))  # 默认关：弱模型校准会压平概率区分度

    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.lgb"

    # Train one continuous target. The same target is used by the portfolio,
    # so model selection is no longer disconnected from backtest economics.
    pred = mat[["future_return", "market"]].copy()
    pred = pred.reset_index()
    models_blob = {"feat_cols": feat_cols, "params": params, "split": seg, "models": {}}
    metrics = {"model_path": str(model_path), "n_features": len(feat_cols), "n_train": {}}
    for target in targets:
        model, rank, blob, n_train = _train_one(feat_cols, params, splits, target, use_ensemble, calibrate)
        pred[target] = mat[target].reset_index(drop=True) if target in mat else np.nan
        pred[f"rank_{target}"] = rank
        models_blob["models"][target] = blob
        metrics["n_train"][target] = n_train
        log.info("[model] %s 训练完成，n_train=%d", target, n_train)

    with open(model_path, "wb") as fh:
        pickle.dump(models_blob, fh)

    # Primary target is used for split marking.
    dates = pd.to_datetime(pred["date"])
    pred["split"] = "unlabeled"
    pred.loc[dates <= pd.Timestamp(seg["train_end"]), "split"] = "train"
    if seg.get("valid_start"):
        v = (dates >= pd.Timestamp(seg["valid_start"])) & (dates <= pd.Timestamp(seg["valid_end"]))
        pred.loc[v, "split"] = "valid"
    if seg.get("test_start"):
        te_end = pd.Timestamp(seg["test_end"]) if seg.get("test_end") else dates.max()
        te = (dates >= pd.Timestamp(seg["test_start"])) & (dates <= te_end)
        pred.loc[te, "split"] = "test"
    pred.loc[pred[targets[0]].isna(), "split"] = "unlabeled"

    write_parquet(pred, "predictions")

    # Quality composite is A-share-specific: yfinance markets intentionally
    # exclude non-PIT financial snapshots, so it must never be forced on US/HK.
    if str(cfg.backtest.get("ranking", "")).lower() == "quality":
        pred["rank_quality"] = quality_signal_rank(mat).to_numpy()
    write_parquet(pred, "predictions")

    # 评估（test + valid，对三个维度分别给出）
    for name in ("valid", "test"):
        seg_df = pred[pred["split"] == name]
        if seg_df.empty:
            continue
        for target in targets:
            sub = seg_df.dropna(subset=[target, f"rank_{target}"])
            if not sub.empty:
                metrics.setdefault(name, {})[target] = evaluate.summarize(
                    sub[f"rank_{target}"], sub[target], sub["date"]
                )

    # Test-set quantile spread against realised raw returns. Because all names
    # share the same daily market component, this long-short spread is also the
    # target residual-return spread.
    test_df = pred[pred["split"] == "test"]
    if not test_df.empty and "future_return" in test_df.columns:
        for target in targets:
            pcol = f"rank_{target}"
            if pcol not in test_df.columns:
                continue
            qa_df = test_df.rename(columns={pcol: "prob"})[["date", "code", "prob", "future_return"]]
            qa = evaluate.quantile_analysis(qa_df, n_quantiles=5)
            if qa:
                metrics.setdefault("test", {}).setdefault(target, {})["quantile"] = qa
                log.info("[model] test 分位多空[%s]: 多空年化=%.3f Sharpe=%.2f 单调性=%.2f",
                         target, qa.get("long_short_ann", float("nan")),
                         qa.get("long_short_sharpe", float("nan")), qa.get("monotonicity", float("nan")))

    log.info("[model] 评估: %s", {k: v for k, v in metrics.items() if k in ("valid", "test")})

    # 持久化「最新一天」快照到 quant/state/（小文件、入库），供每 2h 的 report 复用模型、免重训
    try:
        import shutil

        state_dir = Path(cfg.paths.output_dir).parent.parent / "state"  # quant/data/output -> quant/state
        state_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_path, state_dir / "model.lgb")
        last_date = pred["date"].max()
        pred[pred["date"] == last_date].to_parquet(state_dir / "predictions_latest.parquet", index=False)
        feats_all = read_parquet("features")
        if not feats_all.empty:
            feats_all[feats_all["date"] == str(last_date)].to_parquet(
                state_dir / "features_latest.parquet", index=False
            )
        log.info("[model] state 快照已保存到 %s", state_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning("[model] 保存 state 快照失败: %s", exc)

    return metrics
