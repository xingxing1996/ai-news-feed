"""标签（设计文档第 8 节）。

输出三个维度的二分类 label，对应日报三列概率：
  1. label          未来 horizon 日**相对行业**超额收益是否 > 0（跑赢同行）
  2. abs_label      未来 horizon 日**绝对收益**是否 > 0（会不会涨）
  3. bench_label    未来 horizon 日**相对大盘（同市场等权）**超额收益是否 > 0（跑赢大盘）

  future_return   = close[t+H] / close[t] - 1
  industry_future = 同行业所有股票 future_return 的截面均值（同一 t）
  bench_future    = 同市场（cn/hk/us...）所有股票 future_return 的截面均值（同一 t）

注意：label 用到未来数据，仅用于训练/评估，不能进特征。
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def compute_labels(daily: pd.DataFrame, universe: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """返回 (date, code) 索引：future_return / industry_excess / label / abs_label / bench_label。"""
    if daily.empty:
        return pd.DataFrame()

    ind_map = universe.set_index("code")["industry"].to_dict()
    mkt_map = universe.set_index("code")["market"].to_dict()
    d = daily[["date", "code", "close"]].copy()
    d["industry"] = d["code"].map(ind_map)
    d["market"] = d["code"].map(mkt_map).fillna("other")
    d = d.sort_values(["code", "date"]).reset_index(drop=True)

    # 未来收益（每只股票）
    d["future_return"] = d.groupby("code")["close"].transform(
        lambda c: c.shift(-horizon) / c - 1
    )

    # 行业未来收益截面均值（按 [date, market, industry] 分组：CN/HK/US/KR 各自算行业基准，
    # 避免一个市场的行业被另一个市场同名字的股票稀释）
    ind_fut = d.dropna(subset=["future_return"]).groupby(["date", "market", "industry"])["future_return"].mean()
    # 大盘（同市场等权）未来收益截面均值
    mkt_fut = d.dropna(subset=["future_return"]).groupby(["date", "market"])["future_return"].mean()

    d = d.set_index(["date", "market", "industry"])
    d["industry_future"] = ind_fut
    d = d.reset_index().set_index(["date", "market"])
    d["bench_future"] = mkt_fut
    d = d.reset_index().set_index(["date", "code"]).sort_index()

    d["industry_excess"] = d["future_return"] - d["industry_future"]
    d["bench_excess"] = d["future_return"] - d["bench_future"]
    # 用 nullable Float64 比较，保证 NaN（无未来收益）→ label 为 NA，不被误判为 0
    d["label"] = (d["industry_excess"].astype("Float64") > 0).astype("Int64")
    d["abs_label"] = (d["future_return"].astype("Float64") > 0).astype("Int64")
    d["bench_label"] = (d["bench_excess"].astype("Float64") > 0).astype("Int64")

    for name in ("label", "abs_label", "bench_label"):
        log.info(
            "[label] %s horizon=%d, 正样本比例=%.3f (于有未来收益的样本上)",
            name, horizon,
            float(d[name].dropna().mean()) if d[name].notna().any() else float("nan"),
        )
    return d[["future_return", "industry_excess", "label", "abs_label", "bench_label"]]
