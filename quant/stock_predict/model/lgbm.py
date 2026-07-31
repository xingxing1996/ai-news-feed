"""LightGBM 训练 / 预测。

输出「未来 horizon 日跑赢行业的概率」。
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

_META_COLS = {"future_return", "industry_excess", "industry_excess_neu", "label",
              "industry", "market", "name", "market_cap"}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def _segment_mask(dates: pd.Series, seg: dict, key: str) -> pd.Series:
    start = seg.get(f"{key}_start")
    end = seg.get(f"{key}_end")
    m = pd.Series(True, index=dates.index)
    if start:
        m &= dates >= pd.Timestamp(start)
    if end:
        m &= dates <= pd.Timestamp(end)
    return m


def _split(df: pd.DataFrame, seg: dict) -> dict[str, pd.DataFrame]:
    dates = pd.Series(pd.to_datetime(df.index.get_level_values("date")), index=df.index)
    out = {}
    out["train"] = df[dates <= pd.Timestamp(seg["train_end"])]
    out["valid"] = df[_segment_mask(dates, seg, "valid")]
    if seg.get("test_end"):
        out["test"] = df[_segment_mask(dates, seg, "test")]
    elif seg.get("test_start"):
        out["test"] = df[dates >= pd.Timestamp(seg["test_start"])]
    else:
        out["test"] = df.iloc[0:0]
    return out


def train_and_predict() -> dict:
    cfg = get_settings()
    mat = read_parquet("features")
    if mat.empty:
        raise RuntimeError("features 为空，请先 `stock-predict features`。")
    mat = mat.set_index(["date", "code"]).sort_index()

    feat_cols = _feature_cols(mat)
    log.info("[model] 特征数=%d", len(feat_cols))

    seg = dict(cfg.model.split)
    splits = _split(mat, seg)
    params = dict(cfg.model.lightgbm)

    import lightgbm as lgb  # 延迟导入

    def _xy(df: pd.DataFrame):
        df = df.dropna(subset=["label"])
        X = df[feat_cols]
        y = df["label"].astype(int)
        return X, y

    Xtr, ytr = _xy(splits["train"])
    Xva, yva = _xy(splits["valid"])
    if Xtr.empty:
        raise RuntimeError(f"训练集为空（train_end={seg.get('train_end')}），请放宽切分或采集更多历史。")

    model = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k != "objective"}, objective="binary")
    fit_kwargs = {}
    if not Xva.empty:
        fit_kwargs = {
            "eval_set": [(Xva, yva)],
            "callbacks": [lgb.early_stopping(50, verbose=False)],
        }
    model.fit(Xtr, ytr, **fit_kwargs)

    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.lgb"

    # 预测概率（全集，含 test 与最近用于日报的日期）
    proba_lgb = model.predict_proba(mat[feat_cols])[:, 1]
    models_blob = {"model": model, "feat_cols": feat_cols, "params": params, "split": seg}
    used_ensemble = False
    if bool(cfg.model.get("ensemble", True)):
        try:
            import xgboost as xgb

            xgbm = xgb.XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                n_jobs=-1, eval_metric="logloss", verbosity=0,
            )
            xgbm.fit(Xtr, ytr)
            proba_xgb = xgbm.predict_proba(mat[feat_cols])[:, 1]
            # rank 平均集成（降低单模型方差）
            r1 = pd.Series(proba_lgb).rank(pct=True)
            r2 = pd.Series(proba_xgb).rank(pct=True)
            proba = ((r1 + r2) / 2).values
            models_blob["model_xgb"] = xgbm
            used_ensemble = True
        except Exception as exc:  # noqa: BLE001
            log.info("[model] XGBoost 不可用，仅用 LightGBM：%s", exc)
            proba = proba_lgb
    else:
        proba = proba_lgb

    with open(model_path, "wb") as fh:
        pickle.dump(models_blob, fh)
    pred = mat[["label", "future_return"]].copy() if "label" in mat else mat.iloc[:, 0:0]
    pred["prob"] = proba
    pred = pred.reset_index()

    # 标记 split
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
    pred.loc[pred["label"].isna(), "split"] = "unlabeled"

    write_parquet(pred, "predictions")

    # 评估（test + valid）
    metrics = {"model_path": str(model_path), "n_features": len(feat_cols), "n_train": len(Xtr)}
    for name in ("valid", "test"):
        seg_df = pred[pred["split"] == name].dropna(subset=["label", "prob"])
        if not seg_df.empty:
            metrics[name] = evaluate.summarize(seg_df["prob"], seg_df["label"].astype(int), seg_df["date"])
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
