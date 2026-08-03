# stock-predict 本地部署镜像（A股+港股，AKShare；含 cron 调度 + FastAPI 查询）
# 构建：docker build -t quant-cn .
# 运行(调度+API)：docker run -d -p 8000:8000 -e ARK_API_KEY=xxx -v quant-data:/app/quant/data quant-cn
# 单次训练：    docker run --rm quant-cn run
FROM python:3.10-slim

ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/quant \
    STOCK_PREDICT_CONFIG=/app/quant/config/settings.cn.yaml \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cron tzdata ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 依赖：quant 全量 + API（fastapi/uvicorn）
COPY quant/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt fastapi uvicorn

# 代码 + 配置（.dockerignore 排除 data/.github/缓存）
COPY quant /app/quant
COPY common /app/common
COPY config /app/config
COPY api.py /app/api.py
COPY docker /app/docker
RUN chmod +x /app/docker/*.sh

VOLUME ["/app/quant/data"]
EXPOSE 8000

ENTRYPOINT ["bash", "/app/docker/entrypoint.sh"]
CMD ["serve"]
