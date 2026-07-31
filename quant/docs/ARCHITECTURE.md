# 架构与数据流

## 1. 总览

```
                        ┌─────────────────┐
                        │  CLI (Typer)    │  stock-predict ingest|features|train|backtest|report|run
                        └────────┬────────┘
                                 │
        ┌────────────────────────┼─────────────────────────┐
        ▼                        ▼                         ▼
  data/ (采集+入库)        features/ (因子+label)     model/ + backtest/
        │                        │                         │
        ▼                        ▼                         ▼
  Qlib .bin + Parquet      Alpha158 + 估值/质量        LightGBM → Rank IC
  + SQLite 元数据          + 超额收益 label            → Top-N 组合回测
                                                          │
                                                          ▼
                                                   report/ → AI 投资日报
```

## 2. 数据流（Phase 1）

1. **ingest**
   - `universe.py` 读 `config/universe.yaml` → 写 `stock` 表（SQLite）。
   - `akshare_loader.py` / `yfinance_loader.py` 下载 OHLCV + 财务 → Parquet (`data/raw/`)，
     同时把行情/财务关键列写入 `daily_price` / `financial` 表。
   - `qlib_ingest.py` 把行情转成 Qlib `.bin`（`dump_bin`）→ `data/qlib_data/`，
     并写 `instruments` / `calendars`。
2. **features**
   - `alpha.py`：`qlib.init()` 后用 Alpha158 handler 取价量因子（或 `D.features`）。
   - `valuation.py` / `quality.py`：从 `daily_price`+`financial` 计算**点在时间上正确**的因子，
     关键是估值用滚动历史分位（`pe_percentile` 等）。
   - `labels.py`：`future_return(horizon) − industry_mean_future_return(horizon)` → 二分类 label。
   - 合并为单一 `(datetime, instrument)` 索引的 feature matrix，落盘 Parquet。
3. **train**
   - 按段切分（train/valid/test）。LightGBM 二分类，输出「跑赢行业概率」。
   - `evaluate.py`：Rank IC、ICIR、IC 衰减、命中率（参考，不作主指标）。
4. **backtest**
   - `strategy.py`：每日选概率 Top-N，持有 `hold_days`，带手续费/滑点。
   - `metrics.py`：年化、最大回撤、Sharpe、换手率、超额 vs 基准。
   - 备用路径：把概率分数喂给 Qlib `TopkDropoutStrategy`（见 `qlib_workflow.yaml`）。
5. **report**
   - `explain.py`：对当日 Top 候选，用 SHAP（或规则兜底）取正贡献特征 → 映射成中文理由/风险。
   - `daily.py`：渲染 `report/templates/daily.md.j2` → AI 投资日报。

## 3. 关键设计取舍

- **为什么自研脚本编排而非纯 qrun**：qrun 默认 label 是绝对收益；我们要的是**超额收益 vs 行业**，
  且要叠加自定义估值/质量因子。自研编排保证 label/因子「点在时间上正确」、可控、可解释，
  同时仍复用 Qlib 的数据层、Alpha158 与可选回测。
- **零基建存储**：Qlib `.bin`（行情，列式高性能）+ Parquet（分析层）+ DuckDB（SQL 查询）+ SQLite（元数据）。
  接口均抽象在 `data/warehouse.py` 与 `data/models.py`，未来可换 Postgres/ClickHouse 而不动上层。
- **可解释性优先**：日报的「理由/风险」不是事后编的，而是模型真实依赖的 top 特征（SHAP），
  映射到人话（如 `pe_percentile 低` → 「PE 历史低位」）。规则映射表在 `report/explain.py`。

## 4. 模块清单（关键文件）

| 文件 | 职责 |
|---|---|
| `stock_predict/config.py` | 加载 settings.yaml，解析绝对路径，提供全局 `Settings` 单例 |
| `stock_predict/data/models.py` | SQLAlchemy：stock / daily_price / financial / news |
| `stock_predict/data/warehouse.py` | DuckDB + Parquet 读写；SQLite session |
| `stock_predict/data/akshare_loader.py` | A股 行情/财务（AKShare） |
| `stock_predict/data/yfinance_loader.py` | 港/美股 行情/财务（yfinance） |
| `stock_predict/data/universe.py` | 股票池解析与入库 |
| `stock_predict/data/qlib_ingest.py` | Parquet → Qlib `.bin` |
| `stock_predict/features/alpha.py` | Alpha158 价量因子 |
| `stock_predict/features/valuation.py` | pe/pb/fcf 历史分位 |
| `stock_predict/features/quality.py` | ROE/增长/毛利/FCF |
| `stock_predict/features/industry.py` | 行业周期 industry_score（外部输入接口） |
| `stock_predict/features/labels.py` | 超额收益 label |
| `stock_predict/model/lgbm.py` | LightGBM 训练/保存/预测 |
| `stock_predict/model/evaluate.py` | Rank IC / ICIR |
| `stock_predict/backtest/strategy.py` | Top-N 持有策略 |
| `stock_predict/backtest/metrics.py` | 年化/回撤/Sharpe |
| `stock_predict/report/explain.py` | 特征→理由/风险 |
| `stock_predict/report/daily.py` | AI 投资日报生成 |
| `stock_predict/news/*` | Phase 2：新闻源/embedding/LLM 事件抽取（接口） |
| `stock_predict/agent/*` | Phase 3：投资分析 Agent（接口） |
| `stock_predict/cli.py` | Typer 命令入口 |

## 5. 时间正确性（防未来函数）

- 所有因子只用 `t` 及之前的数据；label 用 `t+1..t+horizon` 的收益。
- 估值分位的窗口只看 `[t-window, t]`，绝不包含未来。
- 财务数据按**报告期 + 发布日**对齐（Phase 1 用发布日近似，财报滞后已在 AKShare 字段体现）。
