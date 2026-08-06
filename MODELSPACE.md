# ModelScope 创空间部署（A股+港股）

把 A股+港股 量化系统部署到 ModelScope 创空间：Gradio 看板 + APScheduler 进程内调度 + `/mnt/workspace` 持久化。

## 文件
- `api.py` — **FastAPI + APScheduler 一体化**（创空间入口，Dockerfile.modelspace 默认跑这个）：REST API + 浏览器看板(`/`) + 进程内调度
- `app.py` — Gradio 版看板（备选；想用 Gradio 就 `CMD python app.py`）
- `config/settings.modelspace.yaml` — A股+港股，数据/模型全走 `/mnt/workspace`
- `requirements-modelspace.txt` — quant 全量 + fastapi/uvicorn + apscheduler(+gradio)
- `Dockerfile.modelspace` — Docker 类型创空间构建（默认跑 FastAPI）

## 关键设计（契合创空间限制）
| 创空间限制 | 对应做法 |
|---|---|
| 重启丢数据，仅 `/mnt/workspace` 持久 | 所有路径（行情/模型/日报/state）→ `/mnt/workspace/...` |
| 无 cron，是 Web app | **APScheduler**（进程内）在 app.py 启动；工作日 17:00 训练 + 每 2h 刷新 |
| 入口是 app（Gradio） | `app.py` 用 Gradio Blocks，`demo.launch(0.0.0.0:7860)` |
| 国内网络 | A股+港股走 AKShare（可用）；美股不在这里（在 GitHub） |

## 部署步骤
1. **创空间新建 Space** → 类型选 **Docker**。
2. 代码：把本仓库推到创空间关联的 git（或上传）。若创空间只认根 `Dockerfile`，把 `Dockerfile.modelspace` 内容作为根 `Dockerfile`（在创空间仓库里）。
3. **端口设 7860**（app.py 监听 7860）。
4. **环境变量**（创空间 Settings）：`ARK_API_KEY=你的ark-key`（新闻 LLM；不配则退化规则）。
5. 启动。首次会后台跑一次训练（A股+港股，约 5–10 分钟），之后：
   - 工作日 17:00 自动全流程训练；
   - 每 2h 复用模型刷新日报。
6. 看板：`推荐`（表格）/ `日报`（markdown）/ `回测` / `状态`（手动触发 + 日志）。

## 自动化单元测试与前置门禁 (API Unit Testing & Pre-push SOP)
系统内置全量 REST API 接口自动化单元测试套件（`quant/tests/test_api.py`），每次修改代码后自动运行单测：

```bash
# 本地运行全量 API 接口单元测试
PYTHONPATH=quant pytest quant/tests/test_api.py -v
```

### 自动化测试覆盖端点
1. `GET /` — 浏览器 HTML 看板（校验 200 响应与 Cache-Control 防死挂响应头）；
2. `GET /recommendations` & `GET /api/recommendations` — 推荐 JSON 数据（校验 `update_time`, `current_price`, `target_price`, `pred_return` 核心字段 100% 存在性）；
3. `GET /report` & `GET /api/report` — Markdown AI 投资日报；
4. `GET /health` & `GET /api/health` — 系统健康与 APScheduler 调度器状态；
5. `GET /backtest` / `GET /files` / `GET /log` / `GET /docs` — 历史回测、文件目录、日志与 Swagger 文档。

---

## 注意
- **数据累积在 `/mnt/workspace/data`**：增量采集，重启后只补最新日，不重拉全量。
- **休眠**：免费 CPU 创空间空闲会休眠，休眠期间 APScheduler 不触发；有人访问/常驻才稳定跑。要 7×24 建议 CPU 常驻型或换云服务器（用根 `Dockerfile`+docker-compose）。
- **美股不在创空间**：美股在 GitHub Actions（`train.yml`），互不影响。
- **模型/数据持久**：训练产物（model.lgb、features、recommendations）都在 `/mnt/workspace`，重启不丢——这点比 GHA（每次新容器）好。

## 本地测试（非创空间）
```bash
# 模拟 /mnt/workspace（本地造个目录）
export OUT_DIR=/tmp/ms/data/output
docker build -f Dockerfile.modelspace -t quant-ms .
docker run --rm -p 7860:7860 -v /tmp/ms:/mnt/workspace -e ARK_API_KEY=xxx quant-ms
# 浏览器 http://localhost:7860
```
