"""特征矩阵构建：把 alpha / valuation / quality / industry / label 合并成一张宽表。

输出 (date, code) 索引的 Parquet：``warehouse/features.parquet``。
其中 label 为 NaN 的行（无未来收益，即最近 horizon 天）仅用于预测，不参与训练。
"""
from __future__ import annotations

import logging

import pandas as pd

from ..config import get_settings
from ..data.models import Financial, get_engine
from ..data.universe import resolve_universe
from ..data.warehouse import read_parquet, write_parquet
from . import alpha, alt, industry, labels, quality, valuation
from .neutralize import add_size_style, neutralize
from .processing import process_features

log = logging.getLogger(__name__)


def _read_financials() -> pd.DataFrame:
    try:
        return pd.read_sql_table(Financial.__tablename__, get_engine())
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def build_feature_matrix() -> tuple[pd.DataFrame, dict]:
    cfg = get_settings()
    daily = read_parquet("daily_price")
    if daily.empty:
        raise RuntimeError("daily_price 为空，请先 `stock-predict ingest`。")

    valuation_df = read_parquet("valuation")
    financial = _read_financials()
    universe = resolve_universe()
    horizon = int(cfg.feature.label_horizon)

    blocks: list[pd.DataFrame] = []

    # 价量因子（alpha）
    alpha_df = alpha.compute_alpha_factors(daily)
    if not alpha_df.empty:
        blocks.append(alpha_df)

    # 估值因子
    if cfg.feature.get("valuation", True) and not valuation_df.empty:
        val_df = valuation.compute_valuation_factors(daily, valuation_df, financial)
        if not val_df.empty:
            blocks.append(val_df)

    # 质量因子
    if cfg.feature.get("quality", True) and not financial.empty:
        q_df = quality.compute_quality_factors(daily, financial)
        if not q_df.empty:
            blocks.append(q_df)

    # 行业因子
    if cfg.feature.get("industry", False):
        ext = cfg.feature.get("industry_external_csv")
        ind_df = industry.compute_industry_factors(daily, universe, ext)
        if not ind_df.empty:
            blocks.append(ind_df)

    # 北向资金因子（A股 另类 alpha）
    if cfg.feature.get("northbound", False):
        nb_df = read_parquet("northbound")
        if not nb_df.empty:
            nb_factors = alt.compute_northbound_factors(nb_df)
            if not nb_factors.empty:
                blocks.append(nb_factors)

    if not blocks:
        raise RuntimeError("未生成任何因子，请检查数据与 feature 配置。")

    mat = pd.concat(blocks, axis=1).sort_index()
    # 去重列名（不同模块可能重名）
    mat = mat.loc[:, ~mat.columns.duplicated()]

    # 标签
    lab = labels.compute_labels(daily, universe, horizon)
    mat = mat.join(lab, how="left")

    # 大盘（同市场等权）未来收益，供日报展示「跑赢大盘」参照
    if "bench_excess" not in mat.columns:
        _mkt = universe.set_index("code")["market"].to_dict()
        _d = daily[["date", "code", "close"]].copy()
        _d["market"] = _d["code"].map(_mkt).fillna("other")
        _d["future_return"] = _d.groupby("code")["close"].transform(
            lambda c: c.shift(-horizon) / c - 1
        )
        _bf = _d.dropna(subset=["future_return"]).groupby(["date", "market"])["future_return"].mean()
        _d = _d.set_index(["date", "market"]).join(_bf.rename("bench_future")).reset_index()
        _d = _d.set_index(["date", "code"])[["bench_future"]]
        mat = mat.join(_d, how="left")
        mat["bench_excess"] = mat["future_return"] - mat["bench_future"]

    # 挂 industry / market 元信息（供截面分组与日报解释）
    meta = universe.set_index("code")[["industry", "market", "name"]]
    mat = mat.join(meta, how="left")

    # 挂 market_cap（用于市值中性化；非特征）
    if "market_cap" in daily.columns:
        mc = daily.set_index(["date", "code"])[["market_cap"]]
        mat = mat.join(mc, how="left")

    # —— 机构级预处理（无未来函数：全部按日截面操作）——
    fcfg = cfg.feature
    feat_cols_now = [c for c in mat.columns if c not in ("future_return", "industry_excess", "industry_excess_neu",
                                                          "label", "abs_label", "bench_label", "bench_excess", "bench_future",
                                                          "industry", "market", "name", "market_cap")]
    # 1) 因子去极值 + 截面标准化
    pmethod = fcfg.get("process", "zscore")
    if pmethod != "none" and feat_cols_now:
        try:
            mat[feat_cols_now] = process_features(mat[feat_cols_now], method=pmethod)
        except Exception as exc:  # noqa: BLE001
            log.warning("[features] 因子预处理失败，用原始值：%s", exc)
    # 2) label 行业+市值中性化：对【连续】超额收益做市值中性，再重新二分类
    if fcfg.get("neutralize", True) and "industry_excess" in mat and "market_cap" in mat:
        try:
            style = add_size_style(mat)
            neu = neutralize(mat["industry_excess"], style)  # 连续超额的中性化残差
            mat["industry_excess_neu"] = neu
            mat["label"] = (neu.astype("Float64") > 0).astype("Int64")  # 残差符号重新二分类
        except Exception as exc:  # noqa: BLE001
            log.warning("[features] label 中性化失败，用原始值：%s", exc)

    # 基本清洗：删除「全部为 NaN 的特征列」
    mat = mat.dropna(axis=1, how="all")

    write_parquet(mat.reset_index(), "features")

    feat_cols = [c for c in mat.columns if c not in ("future_return", "industry_excess", "industry_excess_neu",
                                                       "label", "abs_label", "bench_label", "bench_excess", "bench_future",
                                                       "industry", "market", "name", "market_cap")]
    stats = {
        "rows": int(len(mat)),
        "feature_cols": int(len(feat_cols)),
        "codes": int(mat.index.get_level_values("code").nunique()),
        "dates": int(mat.index.get_level_values("date").nunique()),
        "labeled_rows": int(mat["label"].notna().sum()),
    }
    log.info("[features] 特征矩阵: %s", stats)
    return mat, stats
