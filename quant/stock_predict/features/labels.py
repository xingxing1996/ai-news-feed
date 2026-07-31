"""标签（设计文档第 8 节）。

不要预测涨跌。Label = 未来 horizon 日**相对行业**的超额收益是否 > 0。

  future_return   = close[t+H] / close[t] - 1
  industry_future = 同行业所有股票 future_return 的截面均值（同一 t）
  excess          = future_return - industry_future
  label           = 1 if excess > 0 else 0

注意：label 用到未来数据，仅用于训练/评估，不能进特征。
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def compute_labels(daily: pd.DataFrame, universe: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """返回 (date, code) 索引：future_return / industry_excess / label。"""
    if daily.empty:
        return pd.DataFrame()

    ind_map = universe.set_index("code")["industry"].to_dict()
    d = daily[["date", "code", "close"]].copy()
    d["industry"] = d["code"].map(ind_map)
    d = d.sort_values(["code", "date"]).reset_index(drop=True)

    # 未来收益（每只股票）
    d["future_return"] = d.groupby("code")["close"].transform(
        lambda c: c.shift(-horizon) / c - 1
    )

    # 行业未来收益截面均值
    ind_fut = d.dropna(subset=["future_return"]).groupby(["date", "industry"])["future_return"].mean()
    d = d.set_index(["date", "industry"])
    d["industry_future"] = ind_fut
    d = d.reset_index().set_index(["date", "code"]).sort_index()

    d["industry_excess"] = d["future_return"] - d["industry_future"]
    # 用 nullable Float64 比较，保证 NaN（无未来收益）→ label 为 NA，不被误判为 0
    d["label"] = (d["industry_excess"].astype("Float64") > 0).astype("Int64")

    log.info(
        "[label] horizon=%d, 正样本比例=%.3f (于有未来收益的样本上)",
        horizon,
        float(d["label"].dropna().mean()) if d["label"].notna().any() else float("nan"),
    )
    return d[["future_return", "industry_excess", "label"]]
