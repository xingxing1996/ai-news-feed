# stock-predict 统一镜像：FastAPI(api.py) + APScheduler 进程内调度。
# - ModelScope 创空间：默认 settings.modelspace(数据走 /mnt/workspace)，端口 7860，直接用本 Dockerfile。
# - 本地：用 docker-compose.yml 覆盖 STOCK_PREDICT_CONFIG=settings.cn + 映射端口 + 卷。
FROM python:3.10-slim

ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/quant \
    STOCK_PREDICT_CONFIG=/app/quant/config/settings.modelspace.yaml \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential tzdata ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 依赖：quant 全量 + FastAPI + APScheduler
COPY quant/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt fastapi uvicorn apscheduler

# 代码 + 配置（settings.cn / settings.modelspace 都拷进去，按 env 选）
COPY quant /app/quant
COPY common /app/common
COPY config /app/config
COPY api.py /app/api.py

# 创空间运行时挂载 /mnt/workspace（持久）；本地测试可 -v 或用 docker-compose
EXPOSE 7860
# uvicorn 直接起；api.py 内 APScheduler 负责定时训练/刷新（无需 cron 守护）
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
