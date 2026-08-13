# stock-predict · AI 量化投资研究系统

> 一个 AI 辅助的量化研究系统，覆盖 **A股（核心）/ 港股 / 美股**，每日输出
> 「跑赢行业排名分位 + 评分 + 理由 + 风险」的 AI 投资日报。

基于《AI 量化投资研究系统设计文档 v1.0》落地实现。核心引擎采用
**Microsoft Qlib**，存储层走**零基建本地化**路线（Parquet + DuckDB + SQLite + FAISS），
无需 Docker 即可跑通完整 Phase 1 流程。

---

## 1. 系统能做什么

每日产出形如这样的报告：

```
股票: SK海力士
未来20交易日跑赢行业排名分位: 72%
评分: 85/100

原因:
  + HBM周期改善
  + PE历史低位
  + 资金重新流入

风险:
  - AI资本开支下降
  - 韩国市场流动性风险
```

- **不是**预测涨跌。Label 定义为 `未来20日收益 − 行业收益 > 0`（超额收益）。
- **不是**看准确率。评估看 **Rank IC**（股票排序能力）+ 组合的年化/回撤/Sharpe。
- **不要**「PE<10 就买」。估值用**历史分位**（pe_percentile 等）。

---

## 2. 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| 语言 | Python 3.10 | Qlib 兼容；3.13 会装不上，故用独立 conda 环境 |
| 量化核心 | **Microsoft Qlib** | 数据/因子/回测/组合优化，Alpha158 因子库 |
| 模型 | LightGBM | 第一版不上深度学习；300~500 特征 → 跑赢概率 |
| 数据源 | AKShare（A股）/ yfinance（港美股） | 商业 API（FMP/AlphaVantage/Finnhub）作后备 |
| 行情存储 | Qlib `.bin` + Parquet | 零基建，未来可迁 Postgres/ClickHouse |
| 分析层 | DuckDB | 直接对 Parquet 跑 SQL |
| 元数据 | SQLite（SQLAlchemy） | stock / daily_price / financial / news 表 |
| 向量库 | FAISS（可选） | 新闻 embedding 检索；核心流程非必需，见下文「为啥需要向量」 |
| LLM | **火山方舟 Ark + DeepSeek-V3.2** | 新闻事件抽取（OpenAI 兼容，模型写死） |
| 前端 | FastAPI + 模板（Phase 3） | React 为后续目标 |

---

## 3. 目录结构

```
stock-predict/
├── README.md                  ← 本指引文档
├── docs/ARCHITECTURE.md       ← 架构与数据流详解
├── pyproject.toml             ← 包元数据 + 依赖入口
├── requirements.txt           ← Phase 1 依赖（pip，走清华源）
├── environment.yml            ← conda 环境（python 3.10，走清华源）
├── config/
│   ├── settings.yaml          ← 全局配置：路径/数据源/时间区间
│   ├── universe.yaml          ← 股票池（A股/港股/美股）
│   └── qlib_workflow.yaml     ← Qlib qrun 配置（handler/model/backtest）
├── data/
│   ├── raw/                   ← 原始下载（parquet）
│   ├── qlib_data/             ← Qlib .bin（qlib_dir）
│   ├── warehouse/             ← 分析层 parquet
│   └── meta.db                ← SQLite 元数据
├── stock_predict/             ← Python 包
│   ├── config.py              ← 读 settings.yaml，统一路径
│   ├── cli.py                 ← Typer CLI：ingest/features/train/backtest/report/run
│   ├── data/                  ← 加载器 + Qlib 导入 + 仓库
│   ├── features/              ← Alpha158 + 估值/质量/行业因子 + label
│   ├── model/                 ← LightGBM + Rank IC 评估
│   ├── backtest/              ← Top-N 持有20天 + 组合指标
│   ├── report/                ← 可解释性 + AI 投资日报
│   ├── news/                  ← Phase 2：新闻源/embedding/LLM 事件抽取（接口）
│   └── agent/                 ← Phase 3：投资分析 Agent（接口）
├── scripts/
│   ├── setup_env.sh           ← 建 conda 环境 + 装依赖（清华源）
│   └── run_demo.sh            ← 在演示股票池端到端跑一遍
└── tests/
    └── test_smoke.py          ← import + 数据流冒烟测试
```

---

## 4. 快速开始

### 4.1 建环境（清华镜像 / conda 统一管理）

```bash
cd stock-predict
bash scripts/setup_env.sh           # 建 conda 环境 stock-predict (py3.10) + 装依赖
conda activate stock-predict
```

`setup_env.sh` 已内置清华源：
- pip: `https://pypi.tuna.tsinghua.edu.cn/simple`
- conda: `https://mirrors.tuna.tsinghua.edu.cn/anaconda`

### 4.2 端到端跑一遍（演示股票池）

```bash
bash scripts/run_demo.sh
# 等价于：
stock-predict run --config config/settings.yaml
```

单步执行：

```bash
stock-predict ingest     # 1. 下载行情/财务 → Parquet → Qlib bin
stock-predict features   # 2. Alpha158 + 估值/质量因子 + 超额收益 label
stock-predict train      # 3. LightGBM 训练 + Rank IC 评估
stock-predict backtest   # 4. Top-N 持有20天 → 年化/回撤/Sharpe
stock-predict news       # 5. 新闻采集 → DeepSeek-V3.2 事件抽取 → news_score（可选）
stock-predict report     # 6. 生成 AI 投资日报（含新闻理由）
```

`run` 会按顺序执行以上全部（news 为 best-effort，需网络 + ARK_API_KEY）。

### 4.3 接 LLM（火山方舟 DeepSeek-V3.2）

模型与 endpoint 已写死在 `news/llm_events.py`（Ark OpenAI 兼容 + `deepseek-v3-2-251201`）。
只需提供 API Key（二选一）：

```bash
export ARK_API_KEY=ark-xxxx                 # 方式一：环境变量
# 方式二：写入 config/settings.local.yaml 的 llm.api_key（已 gitignore，不入库）
```

新闻源：A股用 AKShare `stock_news_em`（个股新闻），港美用 RSS（`settings.news.rss_feeds`）。
未配 Key 时自动退化为关键词规则抽取，流水线照跑。

### 4.4 为啥需要向量？（可选）

**核心流程不需要向量。** 当前新闻管线是：
新闻 → 去重 → 关联股票 → DeepSeek 事件抽取 → news_score → 进日报。全程无需 embedding。

向量（FAISS + embedding）只在以下场景才有价值，**按需启用**：
1. **海量新闻语义去重**（标题哈希去不了的同义新闻）；
2. **语义关联股票**（关键词匹配不到的，如「HBM 龙头」→ 海力士）；
3. **RAG**：检索历史上相似事件，给 impact/time_horizon 提供参照。

代码接口已留在 `news/embed.py`（可接你本地的 `qwen3-embedding-0.6B`），默认不启用。

CLI 入口为 `stock-predict`（见 `pyproject.toml [project.scripts]`）。

---

## 5. 设计要点（与设计文档的对应）

| 设计文档要求 | 实现位置 |
|---|---|
| 不重造轮子，Qlib 作核心 | `config/qlib_workflow.yaml` + `features/alpha.py` |
| 估值用历史分位，不是绝对值 | `features/valuation.py`（pe/pb/fcf_yield 分位） |
| 财务质量（ROE/利润增长/FCF） | `features/quality.py` |
| 行业周期 industry_score | `features/industry.py`（外部输入接口，DRAM/HBM 等需另接数据） |
| Label = 超额收益 | `features/labels.py`（future_return − industry_return） |
| 看 Rank IC 而非准确率 | `model/evaluate.py` |
| 组合：Top20 持有20天 | `backtest/strategy.py` + `backtest/metrics.py` |
| 新闻做事件抽取而非情感 | `news/llm_events.py`（Phase 2 接口，输出 direction/impact/time_horizon） |
| 新闻源分层（IR/SEC/RSS/API） | `news/sources.py`（Phase 2 接口） |
| 每日 AI 投资日报 | `report/daily.py` + `report/templates/daily.md.j2` |

---

## 6. 开发路线

- **第一阶段（Phase 1，本仓库已实现）**
  - ✅ 股票池 / 行情数据 / 财务数据
  - ✅ Alpha158 + 自定义因子（100+）
  - ✅ 超额收益 label
  - ✅ LightGBM 模型 + Rank IC
  - ✅ Top-N 回测（年化/回撤/Sharpe）
  - ✅ AI 投资日报

- **第二阶段（Phase 2，新闻事件抽取已通）**
  - ✅ 新闻采集（A股 AKShare 个股新闻 + RSS 港美）
  - ✅ DeepSeek-V3.2 事件抽取 → news_score（接火山方舟 Ark）
  - ✅ 新闻理由接入 AI 投资日报
  - ⬜ 新闻 embedding（FAISS，可选，见「为啥需要向量」）
  - ⬜ news_score 作为特征入模型 / 历史回填回测

- **第三阶段（Phase 3，接口已留）**
  - ⬜ 自动日报调度
  - ⬜ 投资分析 Agent（LangChain）
  - ⬜ 风险监控 / Web Dashboard (React)

---

## 7. 重要说明与风险

1. **Qlib + Python 版本**：Qlib 对 Python 版本敏感，本仓库用独立 `py3.10` 环境。
   基座环境的 3.13 不用于本项目。
2. **数据源可达性**：yfinance 在国内访问可能不稳定（雅虎历史服务调整）；
   A股用 AKShare 稳定；商业 API（FMP/Finnhub）需自行申请 key 放入 `config/settings.yaml`。
3. **industry_score / 行业周期数据**（DRAM 价格、HBM 需求、汽车销量等）没有现成免费源，
   Phase 1 以**外部 CSV 输入接口**形式提供，便于后续接入。
4. **本系统仅为研究用途，不构成投资建议。** 回测过拟合、未来函数、幸存者偏差都需自行复核。

---

## 8. 配置入口

所有可调项集中在 `config/settings.yaml`：
- `paths`: 数据/输出目录
- `universe`: 股票池文件 + 演示规模
- `data`: 数据源开关与回填年限
- `model`: LightGBM 参数、训练/验证/测试切分
- `backtest`: Top-N、持有天数、手续费
- `report`: 输出日报路径与 Top-K
- `llm` / `news`: Phase 2 接入点（OpenAI 兼容 endpoint、embedding 模型路径）

详见 `docs/ARCHITECTURE.md`。
