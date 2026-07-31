"""价量因子（Alpha158 核心）。

主路径：优先用 Qlib 的 Alpha158 handler（需要已 dump 到 .bin）。
兜底路径：pandas 实现 Alpha158 风格的价量因子（~70+ 个），保证无 Qlib 数据也能跑。

设计文档 5.1 市场因子：动量 / 波动 / 最大回撤 / ATR 全部覆盖。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Qlib Alpha158 主路径（best-effort）
# --------------------------------------------------------------------------- #
def _try_qlib_alpha158(start: str, end: str) -> pd.DataFrame | None:
    """尝试用 Qlib Alpha158 取因子。失败返回 None。"""
    try:
        import qlib  # noqa: WPS433
        from qlib.config import REG_CN  # noqa: WPS433
        from qlib.contrib.data.handler import Alpha158  # noqa: WPS433

        from ..config import get_settings

        cfg = get_settings()
        qlib.init(provider_uri=cfg.paths.qlib_dir, region=REG_CN)
        handler = Alpha158(
            start_time=start,
            end_time=end,
            fit_start_time=start,
            fit_end_time=start,  # 我们自己做归一化，这里不 fit
            instruments="all",
            infer_processors=[],
            learn_processors=[],
            label=["Ref($close, -2) / Ref($close, -1) - 1"],
        )
        df = handler.fetch(col_set="feature")
        # instrument 是我们的 qlib symbol（点替换为下划线）；映射回 code
        df = df.rename(index=lambda s: s, level=1)
        df.index = df.index.set_names(["datetime", "instrument"])
        return df
    except Exception as exc:  # noqa: BLE001
        log.info("[alpha] Qlib Alpha158 不可用，使用 pandas 兜底：%s", exc)
        return None


# --------------------------------------------------------------------------- #
# pandas 兜底：Alpha158 风格价量因子
# --------------------------------------------------------------------------- #
_WINS = [5, 10, 20, 30, 60, 120]
_SHORT_WINS = [5, 20, 60]


def _rolling_rank(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(5, w // 3)).rank(pct=True)


def _feats_for_code(g: pd.DataFrame, market_ret: pd.Series) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    c, o, h, l, v = g["close"], g["open"], g["high"], g["low"], g["volume"]
    eps = 1e-12
    out: dict[str, pd.Series] = {}

    # 价格形态
    rng = (h - l).replace(0, eps)
    out["KMID"] = (c - o) / o
    out["KLEN"] = (h - l) / o
    out["KUP"] = (h - np.maximum(o, c)) / o
    out["KLOW"] = (np.minimum(o, c) - l) / o
    out["KMID2"] = (c - o) / rng
    out["OPEN0"] = o / c.shift(1) - 1

    ret = c.pct_change()
    out["RET1D"] = ret

    # 动量 / 均线偏离 / 波动
    for w in _WINS:
        out[f"ROC{w}"] = c.pct_change(w)
        ma = c.rolling(w, min_periods=w // 2).mean()
        out[f"MA{w}"] = c / ma - 1
        out[f"STD{w}"] = c.pct_change().rolling(w, min_periods=w // 2).std()

    # 成交量
    for w in _SHORT_WINS:
        out[f"VSTD{w}"] = v.pct_change().rolling(w, min_periods=w // 2).std()
        out[f"VROC{w}"] = v.pct_change(w)
        out[f"VRANK{w}"] = _rolling_rank(v, w)

    # VWAP 偏离
    for w in [5, 20]:
        vwap = (c * v).rolling(w, min_periods=w // 2).sum() / v.rolling(w, min_periods=w // 2).sum()
        out[f"VWAP{w}"] = c / vwap - 1

    # RSI
    for w in [6, 12, 24]:
        delta = c.diff()
        up = delta.clip(lower=0).rolling(w, min_periods=w // 2).mean()
        down = (-delta.clip(upper=0)).rolling(w, min_periods=w // 2).mean()
        rs = up / (down + eps)
        out[f"RSI{w}"] = 100 - 100 / (1 + rs)

    # 波动率 / 最大回撤 / ATR
    for w in [20, 60]:
        out[f"VOLAT{w}"] = ret.rolling(w, min_periods=w // 2).std() * np.sqrt(252)
    for w in [60, 120]:
        roll_max = c.rolling(w, min_periods=w // 2).max()
        out[f"MDD{w}"] = c / roll_max - 1
    for w in [14, 20]:
        tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        out[f"ATR{w}"] = tr.rolling(w, min_periods=w // 2).mean() / c

    # 历史分位（设计文档强调：用分位而非绝对值）
    for w in [60, 120, 252]:
        out[f"QTLU{w}"] = _rolling_rank(c, w)

    # 相对高低
    for w in [10, 20, 60]:
        out[f"MAXR{w}"] = h.rolling(w, min_periods=w // 2).max() / c - 1
        out[f"MINR{w}"] = l.rolling(w, min_periods=w // 2).min() / c - 1

    # 高阶矩
    for w in [20, 60]:
        out[f"SKEW{w}"] = ret.rolling(w, min_periods=w // 2).skew()
        out[f"KURT{w}"] = ret.rolling(w, min_periods=w // 2).kurt()

    # 与市场的关系（Beta / 相关）
    mkt = market_ret.reindex(g["date"].values).values
    mkt = pd.Series(mkt, index=g.index)
    for w in [20, 60]:
        cov = ret.rolling(w, min_periods=w // 2).cov(mkt)
        var = mkt.rolling(w, min_periods=w // 2).var()
        out[f"BETA{w}"] = cov / (var + eps)
        out[f"CMKT{w}"] = ret.rolling(w, min_periods=w // 2).corr(mkt)

    out_df = pd.DataFrame(out)
    out_df.insert(0, "date", g["date"].values)
    out_df.insert(1, "code", g["code"].values)
    return out_df


def _pandas_alpha(daily: pd.DataFrame) -> pd.DataFrame:
    """pandas 实现的 Alpha158 风格因子。"""
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
    daily["ret"] = daily.groupby("code")["close"].pct_change()
    market_ret = daily.groupby("date")["ret"].mean()

    parts = []
    for code, g in daily.groupby("code"):
        parts.append(_feats_for_code(g, market_ret))
    df = pd.concat(parts, ignore_index=True)
    df = df.set_index(["date", "code"]).sort_index()
    return df


def compute_alpha_factors(daily: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """入口：先试 Qlib Alpha158（需 feature.use_qlib_alpha=true 且已 dump .bin），否则 pandas 兜底。"""
    if daily.empty:
        return pd.DataFrame()

    # Qlib 原生 Alpha158：默认关闭（读取在某些数据上会 hang）；要用 qrun 再开。
    from ..config import get_settings
    use_qlib = bool(get_settings().feature.get("use_qlib_alpha", False)) if hasattr(get_settings(), "feature") else False

    if start is None:
        start = daily["date"].min()
    if end is None:
        end = daily["date"].max()
    qlib_df = _try_qlib_alpha158(str(start), str(end)) if use_qlib else None
    if qlib_df is not None and not qlib_df.empty:
        log.info("[alpha] 使用 Qlib Alpha158，因子数=%d", qlib_df.shape[1])
        # instrument 名（点→下划线）映射回 code
        qlib_df = qlib_df.reset_index()
        qlib_df["instrument"] = qlib_df["instrument"].astype(str)
        qlib_df["code"] = qlib_df["instrument"].str.replace("_", ".", regex=False)
        qlib_df = qlib_df.drop(columns=["instrument"])
        qlib_df["datetime"] = pd.to_datetime(qlib_df["datetime"]).dt.strftime("%Y-%m-%d")
        qlib_df = qlib_df.rename(columns={"datetime": "date"}).set_index(["date", "code"]).sort_index()
        return qlib_df

    df = _pandas_alpha(daily)
    log.info("[alpha] 使用 pandas 兜底，因子数=%d", df.shape[1])
    return df
