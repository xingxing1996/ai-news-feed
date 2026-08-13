"""标签（设计文档第 8 节）。

输出三个维度的二分类 label，对应日报三列排名分位：
  1. label          未来 horizon 日**相对行业**超额收益的截面分位区间（跑赢同行的程度）
  2. abs_label      未来 horizon 日**绝对收益**的截面分位区间（上涨幅度）
  3. bench_label    未来 horizon 日**相对大盘**超额收益的截面分位区间（跑赢大盘的程度）

采用【截面分位区间分类】而非"收益>0"符号分类：
  每日把股票按对应指标排序，top 40% → 1、bottom 40% → 0、中间 20% → NA(不参与训练)。
  这样只学"显著跑赢/跑输同行"的极端组（信噪比远高于">0"，后者把涨0.1%与涨20%等同），
  且保持二分类→排序→日报兼容。无未来收益(最后 horizon 天)→NA。

  future_return   = close[t+H] / close[t] - 1
  industry_future = 同行业所有股票 future_return 的截面均值（同一 t, 按市场分组）
  bench_future    = 同市场（cn/hk/us...）所有股票 future_return 的截面均值（同一 t）

注意：label 用到未来数据，仅用于训练/评估，不能进特征。
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

# 截面分位区间阈值：rank>=ZONE_HI → 1(强者)，rank<=ZONE_LO → 0(弱者)，中间 → NA
ZONE_HI = 0.6
ZONE_LO = 0.4
MIN_INDUSTRY_MEMBERS = 3


def zone_label(s: pd.Series, market: pd.Series | None = None,
               hi: float = ZONE_HI, lo: float = ZONE_LO) -> pd.Series:
    """截面分位区间分类：每日 top(hi 以上)→1、bottom(lo 以下)→0、中间→NA。
    s: (date, code) 索引的连续值 Series。返回 nullable Int64（0/1/NA）。"""
    if market is None:
        rank = s.groupby(level="date").rank(pct=True)
    else:
        market = market.reindex(s.index).fillna("other")
        dates = s.index.get_level_values("date")
        # 市场间币种、交易日和股票池不同，不能放在同一横截面排序。
        rank = s.groupby([dates, market.to_numpy()]).rank(pct=True)
    out = pd.Series(pd.NA, index=s.index, dtype="Float64")
    out[rank >= hi] = 1
    out[rank <= lo] = 0
    return out.astype("Int64")


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
    d["industry_size"] = d.groupby(["date", "market", "industry"])["future_return"].transform("count")
    # 大盘（同市场等权）未来收益截面均值
    mkt_fut = d.dropna(subset=["future_return"]).groupby(["date", "market"])["future_return"].mean()

    d = d.set_index(["date", "market", "industry"])
    d["industry_future"] = ind_fut
    d = d.reset_index().set_index(["date", "market"])
    d["bench_future"] = mkt_fut
    d = d.reset_index().set_index(["date", "code"]).sort_index()

    d["industry_excess"] = d["future_return"] - d["industry_future"]
    d["bench_excess"] = d["future_return"] - d["bench_future"]
    # Primary training target: a continuous, market-neutral forward return.
    # Winsorising limits a few limit-up/down outliers from dominating a
    # regression loss while preserving the return ordering used by the book.
    d["residual_return"] = d["bench_excess"].clip(lower=-0.50, upper=0.50)
    # 行业样本过少时，行业均值会被单只股票自身决定（单成员行业的超额恒为 0）。
    # 这类标的改用市场超额，避免退化标签进入训练。
    sparse_industry = d["industry_size"] < MIN_INDUSTRY_MEMBERS
    d.loc[sparse_industry, "industry_excess"] = d.loc[sparse_industry, "bench_excess"]
    # 截面分位区间分类（top/bottom 40%，中间 20% → NA）：高信噪比，保持二分类→排序→日报兼容
    d["label"] = zone_label(d["industry_excess"].astype("Float64"), d["market"])
    d["abs_label"] = zone_label(d["future_return"].astype("Float64"), d["market"])
    # 不能对 bench_excess 再做同日排序：同一市场当日减去的是同一个 bench_future，
    # 排序会与 future_return 完全相同。这里使用真正的“是否跑赢市场”事件标签。
    d["bench_label"] = pd.Series(pd.NA, index=d.index, dtype="Int64")
    valid_bench = d["bench_excess"].notna()
    d.loc[valid_bench, "bench_label"] = (d.loc[valid_bench, "bench_excess"] > 0).astype("Int64")

    for name in ("label", "abs_label", "bench_label"):
        log.info(
            "[label] %s horizon=%d, 正样本比例=%.3f (于有未来收益的样本上)",
            name, horizon,
            float(d[name].dropna().mean()) if d[name].notna().any() else float("nan"),
        )
    return d[["future_return", "industry_excess", "bench_excess", "residual_return",
              "label", "abs_label", "bench_label"]]
