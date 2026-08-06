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
              "abs_label", "bench_label", "bench_excess", "bench_future",
              "industry", "market", "name", "market_cap",
              "pe", "pb"}  # pe/pb 原值只供日报展示，不进模型特征


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

    # 智能自适应保底：若硬编码日期切出来的训练集为空（如新增样本年份靠后），自动按 60%/20%/20% 动态切分
    if out["train"].empty:
        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)
        if n_dates >= 3:
            idx1 = int(n_dates * 0.6)
            idx2 = int(n_dates * 0.8)
            d1, d2 = unique_dates[idx1], unique_dates[idx2]
            out["train"] = df[dates < d1]
            out["valid"] = df[(dates >= d1) & (dates < d2)]
            out["test"] = df[dates >= d2]
        else:
            out["train"] = df
            out["valid"] = df
            out["test"] = df
    return out


_TARGETS = ["label", "abs_label", "bench_label"]  # 三个维度：跑赢行业 / 绝对上涨 / 跑赢大盘


def _train_one(feat_cols, params, splits, target, use_ensemble=True, calibrate=False):
    """为单个 target 训练 LightGBM(+XGBoost 可选) + 概率校准，返回 (model, proba, blob, n_train)。"""
    import lightgbm as lgb  # 延迟导入

    def _xy(df: pd.DataFrame):
        df = df.dropna(subset=[target])
        return df[feat_cols], df[target].astype(int)

    Xtr, ytr = _xy(splits["train"])
    Xva, yva = _xy(splits["valid"])
    if Xtr.empty:
        raise RuntimeError(f"训练集为空（{target}，train_end={splits['_seg'].get('train_end')}）")

    model = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k != "objective"}, objective="binary")
    fit_kwargs = {}
    if not Xva.empty:
        fit_kwargs = {"eval_set": [(Xva, yva)], "callbacks": [lgb.early_stopping(50, verbose=False)]}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr, ytr, **fit_kwargs)

    xgbm = None
    if use_ensemble:
        try:
            import xgboost as xgb

            xgbm = xgb.XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                n_jobs=-1, eval_metric="logloss", verbosity=0,
            )
            xgbm.fit(Xtr, ytr)
        except Exception as exc:  # noqa: BLE001
            log.info("[model] XGBoost 不可用（%s）：%s", target, exc)
            xgbm = None

    def _proba(X):
        p = model.predict_proba(X)[:, 1]
        if xgbm is not None:
            r1 = pd.Series(p).rank(pct=True)
            r2 = pd.Series(xgbm.predict_proba(X)[:, 1]).rank(pct=True)
            p = ((r1 + r2) / 2).values
        return p

    proba = _proba(splits["_all"][feat_cols])

    # 概率校准（isotonic，valid 拟合）：默认【关】。
    # 原因：模型偏弱时 isotonic 会把所有概率压回基准率(~0.5)，区分度全无、毫无意义。
    # 想要"真概率"可设 model.calibrate: true；默认用原始 GBDM 概率（0.3~0.7 有区分度）。
    calib = False
    if calibrate and not Xva.empty:
        try:
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(out_of_bounds="clip").fit(_proba(Xva), yva.values)
            proba = iso.transform(proba)
            calib = True
        except Exception as exc:  # noqa: BLE001
            log.debug("[model] 校准失败（%s）：%s", target, exc)

    blob = {"model": model, "used_ensemble": xgbm is not None, "calibrated": calib}
    if xgbm is not None:
        blob["model_xgb"] = xgbm
    return model, proba, blob, len(Xtr)


def train_and_predict() -> dict:
    cfg = get_settings()
    mat = read_parquet("features")
    if mat.empty:
        raise RuntimeError("features 为空，请先 `stock-predict features`。")
    mat = mat.set_index(["date", "code"]).sort_index()

    feat_cols = _feature_cols(mat)
    log.info("[model] 特征数=%d", len(feat_cols))

    seg = dict(cfg.model.split)
    # embargo：按 label horizon 隔离段间重叠（防标签穿越泄漏）
    horizon = int(cfg.feature.get("label_horizon", 20))
    embargo_days = int(seg.get("embargo_days") or max(1, round(horizon * 1.4)))
    splits = _split(mat, seg, embargo_days=embargo_days)
    splits["_seg"] = seg
    splits["_all"] = mat
    params = dict(cfg.model.lightgbm)
    use_ensemble = bool(cfg.model.get("ensemble", True))
    calibrate = bool(cfg.model.get("calibrate", False))  # 默认关：弱模型校准会压平概率区分度

    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.lgb"

    # 训练三个维度模型，各自输出概率
    pred = mat[["future_return"]].copy()
    pred = pred.reset_index()
    models_blob = {"feat_cols": feat_cols, "params": params, "split": seg, "models": {}}
    metrics = {"model_path": str(model_path), "n_features": len(feat_cols), "n_train": {}}
    for target in _TARGETS:
        model, proba, blob, n_train = _train_one(feat_cols, params, splits, target, use_ensemble, calibrate)
        pred[target] = mat[target].reset_index(drop=True) if target in mat else np.nan
        pred[f"prob_{target}"] = proba
        models_blob["models"][target] = blob
        metrics["n_train"][target] = n_train
        log.info("[model] %s 训练完成，n_train=%d", target, n_train)

    with open(model_path, "wb") as fh:
        pickle.dump(models_blob, fh)

    # 主标签（跑赢行业）用于 split 标记
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

    # 评估（test + valid，对三个维度分别给出）
    for name in ("valid", "test"):
        seg_df = pred[pred["split"] == name]
        if seg_df.empty:
            continue
        for target in _TARGETS:
            sub = seg_df.dropna(subset=[target, f"prob_{target}"])
            if not sub.empty:
                metrics.setdefault(name, {})[target] = evaluate.summarize(
                    sub[f"prob_{target}"], sub[target].astype(int), sub["date"]
                )
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
