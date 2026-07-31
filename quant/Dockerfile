# stock-predict 容器镜像
# 构建：docker build -t stock-predict .
# 运行（一次性）：docker run --rm -e ARK_API_KEY=xxx -v $PWD/data:/app/data stock-predict
# 定时：见 docker-compose.yml（CRON_SCHEDULE）
FROM python:3.10-slim

ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STOCK_PREDICT_CONFIG=/app/config/settings.yaml

WORKDIR /app

# 系统依赖：build-essential（编译）+ cron（定时）+ ca-certificates/curl/tzdata
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates tzdata cron \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 先装依赖（利用 docker 层缓存）
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝源码与配置（.dockerignore 已排除 data/ settings.local.yaml .git）
COPY stock_predict ./stock_predict
COPY config ./config
COPY scripts ./scripts
COPY tests ./tests
COPY docker ./docker
COPY README.md ./
RUN pip install --no-cache-dir -e . \
    && chmod +x /app/docker/*.sh

# 数据持久化卷
VOLUME ["/app/data"]

ENTRYPOINT ["bash", "/app/docker/entrypoint.sh"]
CMD ["run"]
